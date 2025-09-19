# coding=utf-8
# Copyright 2024 The Efficientids Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layers of item-language interleaved model.

Provides capability of having items and word tokens interleaved in both prefill
and decode.
"""

from typing import Optional

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis.layers import activations as layer_activations
from praxis.layers import attentions
from praxis.layers import base_ops
from praxis.layers import chunk
from praxis.layers import linears
from praxis.layers import transformer_models

import utils


NestedMap = py_utils.NestedMap
LanguageModelType = transformer_models.LanguageModelType
JTensor = pytypes.JTensor
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
AuxLossStruct = base_layer.AuxLossStruct
AUX_LOSS = base_layer.AUX_LOSS
SplitDimsMapping = pytypes.SplitDimsMapping


def log_info(fmt, *args, **kwargs):
  jax.debug.callback(
      lambda *args, **kwargs: logging.info(fmt.format(*args, **kwargs)),
      *args,
      **kwargs,
      ordered=False,
  )


def _compute_z_loss(logits):
  """Returns a z_loss regularization which stablize logits."""
  # Applies stop_gradient to max_logit instead of logits.
  max_logit = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
  exp_x = jnp.exp(logits - max_logit)
  sum_exp_x = jnp.sum(exp_x, axis=-1, keepdims=True)
  log_z = jnp.log(sum_exp_x) + max_logit
  return jnp.square(log_z)


class ItemLanguageSoftmax(base_layer.BaseLayer):
  """A simple softmax layer with word and item cross-entropy outputs.

  Attributes: Attributes copied from embedding_softmax.FullSoftmax
    input_dims: Dimension of the input when used as a softmax layer. This is
    also the depth of the output when used as an embedding layer.
    num_classes: Total number of word target classes/vocabulary.
    soft_cap_logits: If not None logits are soft capped to this value.
    bi_tempered_loss: Not supported.
    label_smoothing_prob: Label smoothing probability.
    label_smoothing_apply_for_eval: If False, disables label smoothing at eval
    time, even if p.label_smoothing_prob > 0. Label smoothing is a form of
    regularization and we may want to disable it at eval time.
    z_loss_weight: If z_loss_weight is nonzero, we add a loss equal to
    z_loss_weight * square(logsumexp(logits, -1))
    bias_init: Init scale (constant) of bias terms.
    feed_forward_tpl: Sub configurable field for the feed-forward layer. If
    None, skip feedforward layer and directly apply softmax to the input.
    scale_before_logits: If set True, activations are scaled with 1/sqrt(M)
    before computing the logits
    chunk_size: chunk size of sequence axis. If set, lax.scan computes softmax
    chunkwise to save HBM. If set, it doesn't return logits and log_probs. 256
    is a compromise between performance and memory usage.  Attributes added for
    item decoding model.
    item_input_dims: Dimension of the item input when used as a softmax layer.
    This is also the depth of the output when used as an embedding layer.
    num_item_classes: Total number of item target classes/vocabulary.
    num_clusters: Total number of item clusters.
    cluster_assignments: A JTensor of shape [num_item_classes] with the cluster
    assignment of each item. Non trainable.
    cluster_indices: A JTensor of shape [num_clusters, num_items_per_cluster]
    with the indices of the items in each cluster. Non trainable.
    in_cluster_id: A JTensor of shape [num_item_classes] with the index within
    cluster of each item. Non trainable.
    cluster_embeddings: A JTensor of shape [num_clusters, item_input_dims] with
    the embeddings of each cluster. Non trainable.
    item_feed_forward_tpl: Sub configurable field for the item feed-forward
    layer. Non trainable.
    item_output_dnn_tpl: Sub configurable field for the dnn layer applied before
    item softmax. If None, skip dnn layer and directly apply item softmax.
    Projects transformer output to item_input_dims.
    einsum_tpl: Sub configurable field for the einsum layer.
    array_lookup_tpl: Sub configurable field for the array lookup layer.
  """

  input_dims: int = 0
  num_classes: int = 0
  soft_cap_logits: float | None = 0.0
  bi_tempered_loss_tpl: LayerTpl | None = template_field(None)
  label_smoothing_prob: float = 0.0
  second_label_smoothing_prob: float = 0.0
  label_smoothing_apply_for_eval: bool = True
  z_loss_weight: float = 0.0
  bias_init: float | None = 0.0
  feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)
  scale_before_logits: bool = False
  chunk_size: int | None = None

  # Item specific attributes.
  full_softmax: bool = False
  item_input_dims: int = 0
  num_item_classes: int = 0
  num_clusters: int = 0
  cluster_assignments: Optional[JTensor] = None
  cluster_indices: Optional[JTensor] = None
  in_cluster_id: Optional[JTensor] = None
  cluster_embeddings: Optional[JTensor] = None
  trainable_cluster_embeddings: bool = False
  item_feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)
  cluster_feed_forward_tpl: LayerTpl = template_field(linears.FeedForward)
  item_output_dnn_tpl: LayerTpl = template_field(linears.MLPBlock)
  item_input_dnn_tpl: LayerTpl | None = template_field(linears.MLPBlock)
  use_item_input_dnn_everywhere: bool = False
  correction: bool = False

  einsum_tpl: LayerTpl = template_field(base_ops.EinsumOp)
  array_lookup_tpl: LayerTpl = template_field(base_ops.ArrayLookup)

  def setup(self):
    self.create_child('einsum', self.einsum_tpl.clone())
    wp = self.weight_split_dims_mapping
    ap = self.activation_split_dims_mapping
    if self.feed_forward_tpl is not None:
      ff_p = self.feed_forward_tpl.clone().set(
          input_dims=self.input_dims,
          output_dims=self.num_classes,
          activation_tpl=pax_fiddle.Config(layer_activations.Identity),
          bias_init=self.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone(),
      )
      self.create_child('logits_ffn', ff_p)
    ff_p_item = self.item_feed_forward_tpl.clone().set(
        input_dims=self.item_input_dims,
        output_dims=self.num_item_classes,
        activation_tpl=pax_fiddle.Config(layer_activations.Identity),
        bias_init=self.bias_init,
        weight_split_dims_mapping=wp.clone(),
        activation_split_dims_mapping=ap.clone(),
    )
    self.create_child('logits_ffn_item', ff_p_item)
    if self.trainable_cluster_embeddings:
      ff_p_cluster = self.cluster_feed_forward_tpl.clone().set(
          input_dims=self.item_input_dims,
          output_dims=self.num_clusters,
          activation_tpl=pax_fiddle.Config(layer_activations.Identity),
          bias_init=self.bias_init,
          weight_split_dims_mapping=wp.clone(),
          activation_split_dims_mapping=ap.clone(),
      )
      self.create_child('logits_ffn_cluster', ff_p_cluster)
    if self.item_output_dnn_tpl is not None:
      self.create_child(
          'item_output_dnn',
          self.item_output_dnn_tpl.clone().set(name='item_output_dnn_tpl'),
      )
    if self.item_input_dnn_tpl is not None:
      self.create_child(
          'item_input_dnn',
          self.item_input_dnn_tpl.clone().set(name='item_input_dnn_tpl'),
      )
    self.create_child('array_lookup', self.array_lookup_tpl.clone())
    if self.bi_tempered_loss_tpl is not None:
      raise ValueError('Bi-tempered loss is not supported.')

  def get_logits_fullsoftmax(
      self, inputs, input_ids = None
  ):
    """Returns logits given the inputs with an option to soft cap it.

    Called during serving fprop and decode_step.

    This is a FullSoftmax to verify the model works as expected for small
    datasets. Scalable serving code is not implemented.

    Args:
      inputs: a single JTensor with shape [batch_size, seq_len, input_dim]
        during fprop and [batch_size, input_dim] during decode_step.
      input_ids: Unused. Needed for API compatibility with downstream usage.

    Returns:
      logits: with shape [..., num_classes+num_item_classes]. Unnormalized
      softmax's logits of word tokens and items.
    """
    del input_ids
    if self.scale_before_logits:
      # activations are scaled with 1/sqrt(input_dims)
      inputs *= self.input_dims**-0.5
    if self.feed_forward_tpl is not None:
      # Compute word token logits.
      word_logits = self.logits_ffn(inputs)

      # Modify the item embeddings to item embeddings + cluster embeddings
      # to match training.
      if (
          self.item_output_dnn_tpl is not None
          and not self.use_item_input_dnn_everywhere
      ):
        inputs = self.item_output_dnn(inputs)
      all_item_embeddings = jnp.asarray(
          jnp.transpose(self.logits_ffn_item.linear.theta.w)
      )
      if (
          self.item_input_dnn_tpl is not None
          and self.use_item_input_dnn_everywhere
      ):
        all_item_embeddings = self.item_input_dnn(all_item_embeddings)
      # [batch_size, num_item_classes]
      all_item_logits = jnp.einsum(
          '...j,ij->...i',
          inputs,
          all_item_embeddings,
      )

      # Set the word logits to -inf to match eval setup of baseline.
      # TODO: Fix this later if serving outside of paper.
      if self.do_eval:
        logits = jnp.concatenate(
            [jnp.ones_like(word_logits) * -jnp.inf, all_item_logits], axis=-1
        )
      else:
        logits = jnp.concatenate([word_logits, all_item_logits], axis=-1)
    else:
      logits = inputs

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
    return logits

  def sum_logits_by_cluster_per_sequence_matrix_mult(self, all_item_logits, cluster_assignments):
    B, S, I = all_item_logits.shape
    num_clusters = self.num_clusters

    # One-hot encode cluster assignments (I, C)
    one_hot_assignments = jax.nn.one_hot(cluster_assignments, num_classes=num_clusters)

    # Reshape for matrix multiplication (B*S, I) @ (I, C) -> (B*S, C)
    reshaped_logits = all_item_logits.reshape(B * S, I)

    # Matrix multiplication to get sums per cluster per sequence
    cluster_sums_per_seq = jnp.dot(reshaped_logits, one_hot_assignments)

    # Reshape back to (B, S, C)
    cluster_sums_per_seq = cluster_sums_per_seq.reshape(B, S, num_clusters)

    # Efficient Replication using Broadcasting
    cluster_sums = cluster_sums_per_seq[:,:, cluster_assignments]  # Broadcasting magic!

    return cluster_sums

  def get_logits(
      self, inputs, input_ids = None
  ):
    """Returns logits given the inputs with an option to soft cap it.

    Called during serving fprop and decode_step.

    This is a FullSoftmax to verify the model works as expected for small
    datasets. Scalable serving code is not implemented.

    Args:
      inputs: a single JTensor with shape [batch_size, seq_len, input_dim]
        during fprop and [batch_size, input_dim] during decode_step.
      input_ids: Unused. Needed for API compatibility with downstream usage.

    Returns:
      logits: with shape [..., num_classes+num_item_classes]. Unnormalized
      softmax's logits of word tokens and items.
    """
    del input_ids
    if self.scale_before_logits:
      # activations are scaled with 1/sqrt(input_dims)
      inputs *= self.input_dims**-0.5
    if self.feed_forward_tpl is not None:
      # Compute word token logits.
      word_logits = self.logits_ffn(inputs)

      # Modify the item embeddings to item embeddings + cluster embeddings
      # to match training.
      # [num_item_classes, item_input_dims]
      if self.trainable_cluster_embeddings:
        cluster_embeddings = jnp.asarray(
            jnp.transpose(self.logits_ffn_cluster.linear.theta.w)
        )
      else:
        cluster_embeddings = jnp.asarray(self.cluster_embeddings)
      all_item_embeddings = jnp.asarray(
          jnp.transpose(self.logits_ffn_item.linear.theta.w)
      )

      if (
          self.item_output_dnn_tpl is not None
          and not self.use_item_input_dnn_everywhere
      ):
        inputs = self.item_output_dnn(inputs)
      if (
          self.item_input_dnn_tpl is not None
          and self.use_item_input_dnn_everywhere
      ):
        all_item_embeddings = self.item_input_dnn(all_item_embeddings)
        cluster_embeddings = self.item_input_dnn(cluster_embeddings)

      cluster_embeddings = self.array_lookup(
          cluster_embeddings,
          jnp.asarray(self.cluster_assignments),
      )
      flattened_item_embeddings = all_item_embeddings + cluster_embeddings
      # [batch_size, num_item_classes]
      all_item_logits = jnp.einsum(
          '...j,ij->...i',
          inputs,
          flattened_item_embeddings,
      )

      # Set the word logits to -inf to match eval setup of baseline.
      # TODO: Fix this later if serving outside of paper.
      if self.correction:
        temp = jnp.einsum(
            '...j,ij->...i',
            inputs,
            all_item_embeddings,
        )
        correction_per_cluster = self.sum_logits_by_cluster_per_sequence_matrix_mult(
            jnp.exp(temp), jnp.asarray(self.cluster_assignments)
        )
        all_item_logits = all_item_logits - jnp.log(correction_per_cluster)
      logits = jnp.concatenate(
          [jnp.ones_like(word_logits) * -jnp.inf, all_item_logits], axis=-1
      )

    else:
      logits = inputs

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
    return logits

  def get_logits_training(
      self,
      inputs,
      cluster_members,
  ):
    """Compute logits and cross-entropy for training.

    Args:
      inputs: a single JTensor with shape [..., input_dim].
      cluster_members: a single JTensor with shape [..., cluster_size] of other
        cluster members of target.

    Returns:
      logits: with shape [..., num_classes+num_clusters]. Unnormalized softmax's
      logits for words and clusters.
      logits_of_cluster_members: with shape [..., cluster_size]. Unnormalized
      softmax's logits for cluster_members.
    """
    if self.scale_before_logits:
      # activations are scaled with 1/sqrt(input_dims)
      inputs *= self.input_dims**-0.5

    # Compute word logits
    if self.feed_forward_tpl is not None:
      logits = self.logits_ffn(inputs)
    else:
      logits = inputs

    # Compute cluster logits
    if self.trainable_cluster_embeddings:
      cluster_embeddings = jnp.asarray(
          jnp.transpose(self.logits_ffn_cluster.linear.theta.w)
      )
    else:
      cluster_embeddings = jnp.asarray(self.cluster_embeddings)
    assert self.item_feed_forward_tpl is not None
    all_item_embeddings = jnp.asarray(
        jnp.transpose(self.logits_ffn_item.linear.theta.w)
    )

    if (
        self.item_output_dnn_tpl is not None
        and not self.use_item_input_dnn_everywhere
    ):
      inputs = self.item_output_dnn(inputs)
    if (
        self.item_input_dnn_tpl is not None
        and self.use_item_input_dnn_everywhere
    ):
      all_item_embeddings = self.item_input_dnn(all_item_embeddings)
      cluster_embeddings = self.item_input_dnn(cluster_embeddings)
    logits_clusters = jnp.einsum(
        '...j,ij->...i',
        inputs,
        cluster_embeddings,
    )
    logits = jnp.concatenate([logits, logits_clusters], axis=-1)

    # Compute item logits
    valid_mask = cluster_members != -1
    cluster_member_embeddings = self.array_lookup(
        all_item_embeddings,
        (cluster_members,),
    )
    logits_of_cluster_members = jnp.where(
        valid_mask,
        self.einsum(
            '...j,...ij->...i',
            inputs,
            cluster_member_embeddings,
        ),
        -jnp.inf,
    )

    # Soft cap logits if applicable.
    if self.soft_cap_logits:
      logits = self.soft_cap_logits * jnp.tanh(logits / self.soft_cap_logits)
      logits_of_cluster_members = self.soft_cap_logits * jnp.tanh(
          logits_of_cluster_members / self.soft_cap_logits
      )
    return logits, logits_of_cluster_members

  def logits_to_logp(self, logits):
    """Converts logits to log probability scores."""
    return jax.nn.log_softmax(logits)

  def __call__(
      self,
      inputs,
      class_weights,
      class_ids = None,
      item_weights = None,
      class_probabilities = None,
      input_ids = None,
  ):
    # pyformat:disable
    """Computes logits, softmax cross entropy etc.

    Args:
      inputs:        [..., input_dims].
      class_weights: [..., 1], weights for each target word.
      class_ids:     [..., 1], int32 type, target labels.
      item_weights: [..., 1], weights to indicate items in target labels.
      class_probabilities: [..., num_classes].
      input_ids: Unused but passed into get_logits. Needed for API compatibility
        with downstream usage.
    Returns:
      A `.NestedMap` containing the following fields

      - logits:    [..., num_classes+num_clusters], unnormalized softmax logits.
      - log_probs: [..., num_classes+num_clusters], normalized softmax logits.
      - per_example_argmax: [...]. argmax of i-th example.
      - per_example_xent:   [...]. Cross entropy between i-th example's
        prediction and its label.
      - per_example_weight: [...]. class_weights casted to this layer's dtype.
      - total_xent:   a scalar, the sum of per_example_weight * per_example_xent.
      - total_weight: a scalar, the sum of per_example_weight.
      - avg_xent: A scalar. total_loss / total_weight.
      - z_loss: (optional) a scalar, the square of logsum logits when
        z_loss_weight > 0.
    """
    # pyformat:enable

    # Assert one of class_ids or class_probabilities is not None
    if class_ids is None and class_probabilities is None:
      raise ValueError('One of class_ids or class_probabilities must be given.')

    chunk_size = self.chunk_size or 0
    use_full_softmax = (
        (chunk_size == 0)
        # The scan optimization is not meaningful, as class_probabilities
        # already allocates full HBM.
        | (class_probabilities is not None)
        # Don't use vmap optimization for small inputs.
        | (inputs.ndim != 3)
        | (inputs.shape[1] < chunk_size)
        # TODO: support scan for z_loss_weight if needed.
        | (self.z_loss_weight > 0.0)
    )
    if use_full_softmax:
      per_example_xent, per_example_argmax, logits, log_probs = (
          self._compute_xent(
              inputs, class_ids, item_weights, class_probabilities
          )
      )
    else:
      per_example_xent, per_example_argmax = self._compute_xent_scan(
          inputs,
          class_ids,
          item_weights,
      )
      logits = log_probs = None

    # Compute total softmax cross-entropy loss for the output tensor.
    total_xent = jnp.sum(
        jnp.expand_dims(per_example_xent, axis=-1) * class_weights,
        dtype=jnp.float32,
    )
    total_weight = jnp.sum(class_weights, dtype=jnp.float32)

    if self.z_loss_weight > 0.0:
      assert logits is not None
      z_loss = (
          jnp.sum(_compute_z_loss(logits) * class_weights, dtype=jnp.float32)
          / total_weight
      )
      z_loss *= self.z_loss_weight
      self.add_summary('aux_z_loss', z_loss)
      self.add_aux_loss('aux_z_loss', z_loss)

    if logits is not None:
      logits = logits.astype(inputs.dtype)
      log_probs = log_probs.astype(inputs.dtype)

    output_nmap = NestedMap(
        logits=logits,
        log_probs=log_probs,
        per_example_argmax=per_example_argmax,
        per_example_xent=per_example_xent.astype(jnp.float32),
        total_xent=total_xent,
        total_weight=total_weight,
        avg_xent=(total_xent / (total_weight + 1e-6)).astype(jnp.float32),
    )
    if class_ids is not None:
      output_nmap.accuracy = (
          jnp.sum(
              (per_example_argmax[Ellipsis, jnp.newaxis] == class_ids)
              * class_weights
          )
          / total_weight
      )
    if self.z_loss_weight > 0.0:
      output_nmap['z_loss'] = z_loss
    return output_nmap

  @nn.nowrap
  def _get_argmax_for_metrics(
      self,
      inputs,
      logits_from_first_pass,
      logits_from_second_pass,
      cluster_members,
  ):
    """Computes per_example_argmax which is later used for computing metrics."""
    logits = self.get_logits(inputs)
    return (
        jax.lax.stop_gradient(jnp.argmax(logits.astype(jnp.float32), axis=-1)),
        jax.lax.stop_gradient(logits),
    )

  @nn.nowrap
  def _compute_xent_fullsoftmax(
      self,
      inputs,
      class_ids = None,
      item_weights = None,
      class_probabilities = None,
  ):
    """Computes logits and softmax cross entropy.

    Args:
      inputs:        [..., input_dims].
      class_ids:     [..., 1], int32 type, target labels.
      item_weights:     [..., 1], float32 type, Indicates class_id is item.
      class_probabilities: [..., num_classes + num_clusters].

    Returns:
      per_example_argmax: [...]. argmax of i-th example.
      per_example_xent:   [...]. Cross entropy between i-th example's
        prediction and its label.
      logits:    [..., num_classes+num_clusters], unnormalized softmax logits.
      log_probs: [..., num_classes+num_clusters], normalized softmax logits.
    """

    logits = self.get_logits_fullsoftmax(inputs)

    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)
    # Update class_ids for items.
    class_ids = (
        item_weights * (class_ids + self.num_classes)
        + (1 - item_weights) * class_ids
    )

    # Calculate xent for first pass.
    if class_probabilities is None:
      first_pass_num_classes = self.num_classes + self.num_item_classes
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(class_ids, axis=-1),
          first_pass_num_classes,
          dtype=jnp.float32,
      )
      if self.label_smoothing_prob > 0.0:
        # Label smoothing reduce the probability of the label from 1 to
        # 1 - label_smoothing_prob, and redistribute label_smoothing_prob to the
        # rest of first_pass_num_classes - 1 classes where each class has a
        # probability of label_smoothing_prob / (first_pass_num_classes - 1).
        if not self.do_eval or self.label_smoothing_apply_for_eval:
          # We may want to disable label smoothing at eval time.
          other_prob = self.label_smoothing_prob / (first_pass_num_classes - 1)
          class_probabilities = (
              (1.0 - self.label_smoothing_prob) * class_probabilities
              + other_prob * (1.0 - class_probabilities)
          ).astype(jnp.float32)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)
    per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32
    )

    def add_custom_summary(x, weight, name):
      x_sum = jnp.sum(x * weight) / jnp.sum(weight)
      self.add_summary(name, x_sum)

    add_custom_summary(
        jnp.expand_dims(per_example_xent, axis=-1),
        item_weights,
        'first_pass_xent',
    )

    add_custom_summary(
        jnp.mean(logits, axis=-1, keepdims=True),
        item_weights,
        'logitspre_items',
    )

    # For eval, we use full softmax only on items.
    if self.do_eval:
      logits = jax.lax.stop_gradient(logits)
      per_example_xent = jax.lax.stop_gradient(per_example_xent)
    per_example_argmax = jax.lax.stop_gradient(
        jnp.argmax(logits.astype(jnp.float32), axis=-1)
    )
    add_custom_summary(
        jnp.expand_dims(per_example_argmax, axis=-1),
        item_weights,
        'per_example_argmax',
    )
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
    return per_example_xent, per_example_argmax, logits, log_probs

  @nn.nowrap
  def _compute_xent(
      self,
      inputs,
      class_ids = None,
      item_weights = None,
      class_probabilities = None,
  ):
    """Computes logits and softmax cross entropy.

    Args:
      inputs:        [..., input_dims].
      class_ids:     [..., 1], int32 type, target labels.
      item_weights:     [..., 1], float32 type, Indicates class_id is item.
      class_probabilities: [..., num_classes + num_clusters].

    Returns:
      per_example_argmax: [...]. argmax of i-th example.
      per_example_xent:   [...]. Cross entropy between i-th example's
        prediction and its label.
      logits:    [..., num_classes+num_clusters], unnormalized softmax logits.
      log_probs: [..., num_classes+num_clusters], normalized softmax logits.
    """

    if self.full_softmax:
      return self._compute_xent_fullsoftmax(
          inputs, class_ids, item_weights, class_probabilities
      )

    item_cluster_ids = jnp.take_along_axis(
        self.cluster_assignments,
        jnp.reshape(class_ids, [-1]),
        axis=0,
    )
    cluster_members = jnp.take_along_axis(
        self.cluster_indices,
        jnp.expand_dims(item_cluster_ids, [-1]),
        axis=0,
    )
    item_cluster_ids = jnp.reshape(
        item_cluster_ids,
        [np.shape(class_ids)[0], jnp.shape(class_ids)[1], -1],
    )
    cluster_members = jnp.reshape(
        cluster_members,
        [np.shape(class_ids)[0], jnp.shape(class_ids)[1], -1],
    )
    logits, logits_of_cluster_members = self.get_logits_training(
        inputs, cluster_members
    )

    # We perform softmax in float32 to improve stability.
    logits = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits)
    logits_of_cluster_members = logits_of_cluster_members.astype(jnp.float32)
    # Invalid cluster members of value -1 are ignored
    members_log_probs = jax.nn.log_softmax(
        logits_of_cluster_members, where=jnp.not_equal(cluster_members, -1)
    )
    members_log_probs = jnp.where(
        jnp.not_equal(cluster_members, -1), members_log_probs, 0.0
    )

    # Update class_ids for items to the item cluster id.
    first_pass_class_ids = (
        item_weights * (item_cluster_ids + self.num_classes)
        + (1 - item_weights) * class_ids
    )

    # Calculate xent for first pass.
    if class_probabilities is None:
      first_pass_num_classes = self.num_classes + self.num_clusters
      class_probabilities = jax.nn.one_hot(
          jnp.squeeze(first_pass_class_ids, axis=-1),
          first_pass_num_classes,
          dtype=jnp.float32,
      )
      if self.label_smoothing_prob > 0.0:
        # Label smoothing reduce the probability of the label from 1 to
        # 1 - label_smoothing_prob, and redistribute label_smoothing_prob to the
        # rest of first_pass_num_classes - 1 classes where each class has a
        # probability of label_smoothing_prob / (first_pass_num_classes - 1).
        if not self.do_eval or self.label_smoothing_apply_for_eval:
          # We may want to disable label smoothing at eval time.
          other_prob = self.label_smoothing_prob / (first_pass_num_classes - 1)
          class_probabilities = (
              (1.0 - self.label_smoothing_prob) * class_probabilities
              + other_prob * (1.0 - class_probabilities)
          ).astype(jnp.float32)
      class_probabilities = jax.lax.stop_gradient(class_probabilities)
    per_example_xent = -jnp.sum(
        log_probs * class_probabilities, axis=-1, dtype=jnp.float32
    )

    # Calculate class_probabilities for second pass.
    item_class_probabilities = jnp.where(
        jnp.equal(cluster_members, class_ids),
        jnp.ones_like(cluster_members),
        jnp.zeros_like(cluster_members),
    )
    valid_mask = jnp.where(
        jnp.not_equal(cluster_members, -1.0),
        jnp.ones_like(cluster_members),
        jnp.zeros_like(cluster_members),
    )
    second_pass_items_size = jnp.sum(valid_mask, axis=-1, keepdims=True)
    # second_pass_items_size = jnp.shape(cluster_members)[-1]
    # Calculate xent for second pass.
    if self.second_label_smoothing_prob > 0.0:
      # Label smoothing reduce the probability of the label from 1 to
      # 1 - second_label_smoothing_prob, and redistribute second_label_smoothing_prob to the
      # rest of second_pass_items_size - 1 classes where each class has a
      # probability of second_label_smoothing_prob / (second_pass_items_size - 1).
      if not self.do_eval or self.label_smoothing_apply_for_eval:
        # We may want to disable label smoothing at eval time.
        other_prob = self.second_label_smoothing_prob / (
            second_pass_items_size - 1
        )
        item_class_probabilities = (
            (1.0 - self.second_label_smoothing_prob) * item_class_probabilities
            + other_prob * (1.0 - item_class_probabilities)
        ).astype(jnp.float32)
    item_class_probabilities = jax.lax.stop_gradient(item_class_probabilities)
    second_per_example_xent = -jnp.sum(
        members_log_probs * item_class_probabilities,
        axis=-1,
        dtype=jnp.float32,
    )
    second_per_example_xent = second_per_example_xent * jnp.squeeze(
        item_weights, -1
    )

    def add_custom_summary(x, weight, name):
      x_sum = jnp.sum(x * weight) / jnp.sum(weight)
      self.add_summary(name, x_sum)

    add_custom_summary(
        jnp.expand_dims(per_example_xent, axis=-1),
        item_weights,
        'first_pass_xent',
    )
    add_custom_summary(
        jnp.expand_dims(second_per_example_xent, axis=-1),
        item_weights,
        'second_pass_xent',
    )

    # Add first and second pass xent.
    per_example_xent = per_example_xent + second_per_example_xent
    per_example_argmax, logits = self._get_argmax_for_metrics(
        inputs, logits, logits_of_cluster_members, cluster_members
    )
    add_custom_summary(
        jnp.mean(logits, axis=-1, keepdims=True), item_weights, 'logits_items'
    )
    add_custom_summary(
        jnp.mean(logits_of_cluster_members, axis=-1, keepdims=True),
        item_weights,
        'logits_of_cluster_members',
    )
    add_custom_summary(
        jnp.expand_dims(per_example_argmax, axis=-1),
        item_weights,
        'per_example_argmax',
    )
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
    return per_example_xent, per_example_argmax, logits, log_probs

  @nn.nowrap
  def _compute_xent_scan(
      self,
      inputs,
      class_ids,
      item_weights = None,
  ):
    """Computes softmax cross entropy using scan.

    Args:
      inputs:        [B, L, input_dims].
      class_ids:     [B, L, 1], int32 type, target labels.
      item_weights:     [B, L, 1], float32 type, Indicates class_id is item.

    Returns:
      per_example_xent: [B, L]. Cross entropy between i-th example's
      per_example_argmax: [B, L]. argmax of i-th example.
    """

    def step_fn(_, step_inputs):
      inputs, class_ids, item_weights = step_inputs
      per_example_xent, per_example_argmax, _, _ = self._compute_xent(
          inputs, class_ids, item_weights
      )
      return None, (per_example_xent, per_example_argmax)

    seqlen = inputs.shape[1]
    input_chunk = chunk.chunk(inputs, chunk_size=self.chunk_size)
    class_ids = chunk.chunk(class_ids, chunk_size=self.chunk_size)
    item_weights = chunk.chunk(item_weights, chunk_size=self.chunk_size)
    # Workaround: flax lazy init doesn't work in scan, so init all sublayers.
    step_fn(None, (input_chunk[0, :1], class_ids[0, :1], item_weights[0, :1]))
    _, (per_example_xent, per_example_argmax) = jax.lax.scan(
        jax.remat(step_fn), None, (input_chunk, class_ids, item_weights)
    )
    per_example_xent = chunk.unchunk(per_example_xent, seqlen=seqlen)
    per_example_argmax = chunk.unchunk(per_example_argmax, seqlen=seqlen)
    return per_example_xent, per_example_argmax


class SharedEmbeddingSoftmax(ItemLanguageSoftmax):
  """A wrapper for ItemLanguageSoftmax.

  This is to support any calls to embedding_softmax.SharedEmbeddingSoftmax from
  the model.

  Attributes:
    lookup_style: Style of lookup, one of index or matmul.
    scale_sqrt_depth: If set True, activations are scaled with
      sqrt(embedding_dim) in emb_lookup.
  """

  lookup_style: str = 'index'
  scale_sqrt_depth: bool = False

  class ActivationSharding(base_layer.BaseLayer.ActivationSharding):
    """Represents how intermediate values should be partitioned across a mesh.

    Attributes:
      emb_out_split_dims_mapping: Sharding of the emb output.
      extend_step_out: Sharding annotations for the primary extend step output.
    """

    emb_out_split_dims_mapping: SplitDimsMapping = None
    extend_step_out: SplitDimsMapping = None

  def emb_lookup(self, ids):
    ap = self.activation_split_dims_mapping
    emb_var = jnp.transpose(self.logits_ffn.linear.theta.w)
    if self.lookup_style == 'index':
      embs = self.array_lookup(jnp.asarray(emb_var), (ids,))
    elif self.lookup_style == 'matmul':
      # Explicit casting to fprop_dtype needed for bf16.
      one_hot_ids = jax.nn.one_hot(
          ids, self.num_classes, dtype=self.fprop_dtype
      )
      embs = self.einsum('...y,yz->...z', one_hot_ids, emb_var)
    else:
      raise ValueError('Unknown lookup style.')
    # Scale with sqrt(embedding dims)
    if self.scale_sqrt_depth:
      embs *= self.input_dims**0.5

    embs = base_layer.maybe_shard(
        embs, ap.emb_out_split_dims_mapping, self.mesh_axis_names
    )
    return embs

  def extend_step(self, ids, *, time_step):
    del time_step  # Not used.
    return self.emb_lookup(ids)


class InterleavedTransformerLm(transformer_models.TransformerLm):
  """Transformer language model supporting item encoded inputs and outputs.

  Attributes:
    item_input_dnn_tpl: Simple DNN item encoder projects item to language token
      space.
  """
  max_num_items: int = 10
  trainable_item_embeddings: bool = False
  trainable_input_only_embeddings: bool = False

  def setup(self):
    assert self.ngrammer_tpl is None
    super().setup()

  def _compute_softmax_loss(
      self,
      activations,
      labels,
      item_weights = None,
      input_ids = None,
  ):
    """Computes cross entropy loss."""
    class_ids = None
    class_probabilities = None
    if 'class_ids' in labels:
      class_ids = labels.class_ids[:, :, jnp.newaxis]
    if 'class_probabilities' in labels:
      class_probabilities = labels.class_probabilities
    class_weights = labels.class_weights[:, :, jnp.newaxis]
    extra_kw_args = (
        {'input_ids': input_ids[Ellipsis, jnp.newaxis]}
        if input_ids is not None
        else {}
    )
    return self.softmax(
        activations,
        class_weights,
        class_ids=class_ids,
        item_weights=item_weights,
        class_probabilities=class_probabilities,
        **extra_kw_args,
    )

  def compute_loss(
      self,
      activations,
      labels = None,
      item_weights = None,
      input_ids = None,
  ):
    """Computes cross entropy loss.

    Args:
      activations: Output of last layer of shape [B, T, D].
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities. item_weights
        , a JTensor with shape [B, T] of float dtype indicating if the class_id
        is an item.
      input_ids: The input ids to the model of shape [B, T].

    Returns:
      Returns xent_output, where `xent_output` is a `.NestedMap` as defined by
      `SoftmaxLayer`'s return. In addition, per_sequence_xent is added which
      equal to the sum of xent loss for tokens in a sequence.
    """
    if labels is None:
      extra_kw_args = {'input_ids': input_ids} if input_ids is not None else {}
      logits = self.softmax.get_logits(inputs=activations, **extra_kw_args)
      xent_output = NestedMap(logits=logits)
      # For numerical stability, use fp32 for softmax and log_softmax.
      logits_dtype = logits.dtype
      casted_logits = logits.astype(jnp.float32)
      xent_output.log_probs = jax.nn.log_softmax(casted_logits).astype(
          logits_dtype
      )
      xent_output.probs = jax.nn.softmax(casted_logits).astype(logits_dtype)
    else:
      class_ids = None
      class_probabilities = None
      if 'class_ids' in labels:
        class_ids = labels.class_ids[:, :, jnp.newaxis]
      if 'class_probabilities' in labels:
        class_probabilities = labels.class_probabilities
      class_weights = labels.class_weights[:, :, jnp.newaxis]
      extra_kw_args = (
          {'input_ids': input_ids[Ellipsis, jnp.newaxis]}
          if input_ids is not None
          else {}
      )
      xent_output = self.softmax(
          activations,
          class_weights,
          class_ids=class_ids,
          item_weights=item_weights,
          class_probabilities=class_probabilities,
          **extra_kw_args,
      )
      per_token_xent = xent_output.per_example_xent * labels.class_weights
      xent_output.per_token_xent = per_token_xent
      xent_output.per_sequence_xent = jnp.sum(per_token_xent, -1)

      # Sum aux_loss and add to avg_xent.
      xent_output.total_loss = xent_output.avg_xent

      # Add entropy loss if entropy_loss_weight is not None
      if self.entropy_loss_weight is not None:
        per_token_entropy_loss = jnp.sum(
            jax.nn.softmax(xent_output.logits)
            * jax.nn.log_softmax(xent_output.logits),
            axis=-1,
        )
        avg_entropy_loss = jnp.sum(
            per_token_entropy_loss * labels.class_weights
        ) / jnp.sum(labels.class_weights)
        self.add_summary(
            'per_token_logits',
            jnp.mean(xent_output.logits, -1)
            * labels.class_weights
            / jnp.sum(labels.class_weights),
        )
        xent_output.avg_entropy_loss = avg_entropy_loss
        self.add_summary('avg_entropy_loss', avg_entropy_loss)
        xent_output.total_loss += (
            xent_output.avg_entropy_loss * self.entropy_loss_weight
        )

      if not self.skip_aux_loss:
        aux_loss = 0.0
        aux_loss_weight = 0.0
        if AUX_LOSS in self.variables:
          aux_loss_values = jax.tree_util.tree_leaves(
              self.variables[AUX_LOSS],
              is_leaf=lambda x: isinstance(x, AuxLossStruct),
          )
          for v in aux_loss_values:
            assert isinstance(v, AuxLossStruct)
            aux_loss += jnp.sum(v.value)
            aux_loss_weight += jnp.sum(v.weight)
        if not isinstance(aux_loss, jnp.ndarray):
          aux_loss = jnp.array(aux_loss, dtype=self.fprop_dtype)
          aux_loss_weight = jnp.array(aux_loss_weight, dtype=self.fprop_dtype)
        self.add_summary('total_aux_loss', aux_loss)
        self.add_summary('total_aux_loss_weight', aux_loss_weight)
        xent_output.aux_loss = aux_loss
        xent_output.aux_loss_weight = aux_loss_weight
        # This is the loss to minimize.
        xent_output.total_loss += xent_output.aux_loss

    # Also output the activations, so that if necessary, we can attach a pooler
    # on top of the activations.
    #
    # NOTE: Recording activations here is a bit awkward yet unfortunately
    # necessary, as we often need both activations and logits. The current
    # implementation returns either activations or logits based on
    # `skip_compute_loss` which is fixed at the module level, and it is
    # tricky to first compute the activation and then logits which would require
    # extensive modification to `LanguageModel.compute_loss` (and other
    # callers).
    if self.record_activations_in_xent_output:
      xent_output.activations = activations
    return xent_output

  def _prepare_input(
      self,
      inputs,
      paddings,
      segment_pos = None,
      item_weights = None,
      **input_kwargs,
  ):
    del input_kwargs
    seq_length = inputs.shape[1]

    # Get the input embeddings.
    if self.separate_embedding_tpl is not None:
      input_emb = self.embedding_lookup.emb_lookup(inputs)
    else:
      input_emb = self.softmax.emb_lookup(inputs)

    # Interleave items and language inputs
    item_embeddings = self.softmax.array_lookup(
        jnp.asarray(jnp.transpose(self.softmax.logits_ffn_item.linear.theta.w)),
        (jnp.reshape(inputs, [-1]),),
    )
    if self.softmax.item_input_dnn_tpl is not None:
      item_embeddings = self.softmax.item_input_dnn(item_embeddings)
    item_embeddings = jnp.reshape(
        item_embeddings,
        jnp.shape(input_emb),
    )
    input_emb = jnp.where(item_weights, item_embeddings, input_emb)

    # Add NGrammer to the source embeddings.
    if self.ngrammer_tpl is not None:
      if self.separate_embedding_tpl is not None:
        emb_var = self.embedding_lookup.theta.emb_var
      else:
        if hasattr(self.softmax, 'logits_ffn'):
          emb_var = jnp.transpose(self.softmax.logits_ffn.linear.theta.w)
        else:
          # For the class where its difference from original
          # SharedEmbeddingSoftmax is that
          # - has its own weight ('w'), not through logits_ffn.
          # - 'w' already has num_class as first dimensions so don't transpose.
          emb_var = self.softmax.theta.w
      input_emb = self.ngrammer(
          input_ids=inputs,
          input_embs=input_emb,
          paddings=paddings,
          segment_pos=segment_pos,
          emb_var=emb_var,
      )

    if self.position_emb_tpl is not None:
      position_emb = self.position_emb(
          seq_length=seq_length, position=segment_pos
      )
      inputs = jnp.add(input_emb, position_emb)
    else:
      inputs = input_emb
    return inputs

  def __call__(
      self,
      inputs,
      paddings,
      labels = None,
      input_item_weights = None,
      label_item_weights = None,
      segment_ids = None,
      segment_pos = None,
      causal_attention_mask = None,
      segment_mask = None,
      start_time_step = 0,
      **input_kwargs,
  ):
    """Computes xent loss given the language model inputs.

    Args:
      inputs: Input ids. An int32 JTensor of shape [B, T].
      paddings: A 0/1 JTensor of shape [B, T] with 1 denoting padding.
      labels: A `.NestedMap` containing the following fields: class_weights, a
        JTensor with shape [B, T] containing weights for each target word.
        class_ids, a JTensor with shape [B, T] of int32 dtype containing the
        target class labels. class_probabilities, a JTensor with shape [B, T, V]
        of float values indicating class-membership probabilities.
      segment_ids: A JTensor of shape [B, T]. The segment that each token
        belongs to.
      segment_pos: A JTensor of shape [B, T]. The position of each token in a
        segment.
      causal_attention_mask: A JTensor of shape [B, T] where 1 indicates a token
        position with causal attention and 0 indicates bidirectional attention.
        This overrides part of the causal mask.
      segment_mask: Optional pre-defined segment_mask passed to the transformer.
        A JTensor of shape [B, 1, T, T]. If it is None, the segment_mask will be
        inferred from the LanguageModelType `model_type` hparam.
      start_time_step: Decode extend_step start time step. When decoding after
        prefix, start_time_step will be prefix_len.
      **input_kwargs: additional input kwargs to be sent to the transformer.

    Returns:
      Returns xent_output, where
      `xent_output` is a `.NestedMap` as defined by `SoftmaxLayer`'s return. In
      addition, per_sequence_xent is added which equal to the sum of xent loss
      for tokens in a sequence.
    """

    batch, seq_length = inputs.shape[:2]

    paddings_float32 = paddings.astype(jnp.float32)
    num_unpadded_tokens = jnp.sum(1.0 - paddings_float32)
    self.add_summary('num_unpadded_tokens', num_unpadded_tokens)
    if inputs.size != 0:
      num_tokens = jnp.array(inputs.size, jnp.float32)
      ratio_unpadded_tokens = num_unpadded_tokens / num_tokens
      self.add_summary('ratio_unpadded_tokens', ratio_unpadded_tokens)

    if segment_ids is None:
      assert segment_pos is None
      # Fold the paddings with the segment mask
      segment_ids = jnp.asarray(1 - paddings, jnp.int32)
      segment_pos = jnp.tile(
          jnp.arange(seq_length, dtype=jnp.int32)[None, :], [batch, 1]
      )
    input_ids = inputs
    inputs = self._prepare_input(
        inputs,
        paddings,
        segment_pos=segment_pos,
        item_weights=jnp.expand_dims(input_item_weights, -1),
        **input_kwargs,
    )

    if segment_mask is None:
      if self.model_type == LanguageModelType.BIDIRECTIONAL:
        segment_mask = attentions.segment_mask(
            segment_ids, segment_ids, inputs.dtype
        )
      else:
        segment_mask = attentions.causal_segment_mask(
            segment_ids, inputs.dtype, causal_attention_mask
        )

    self.update_decode_state('time_step', start_time_step)  # pytype: disable=wrong-arg-types  # jax-ndarray
    output = self.transformer(
        inputs, paddings, segment_mask=segment_mask, segment_pos=segment_pos
    )

    # Final layer norm
    if self.final_ln_tpl is not None:
      output = self.final_ln(output)

    if self.skip_compute_loss:
      return output
    else:
      return self.compute_loss(
          output,
          labels,
          jnp.expand_dims(label_item_weights, -1),
          input_ids=input_ids,
      )
