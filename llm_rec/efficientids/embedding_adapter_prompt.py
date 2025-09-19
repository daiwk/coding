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

from typing import Any, Union

import jax
from jax import numpy as jnp
from praxis import base_layer
from praxis import base_model
from praxis import pax_fiddle
from praxis.layers import linears
from praxis.layers import models
from praxis.layers import transformers

import metrics
import utils
import interleaved_transformer_lm


LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
WeightInit = base_layer.WeightInit
NestedMap = models.NestedMap
WeightedScalars = base_model.WeightedScalars
JTensor = base_layer.JTensor
template_field = pax_fiddle.template_field


def _get_item_weights(weights: JTensor, item_indices: JTensor) -> JTensor:
  """Returns item weights for labels."""
  placeholder_weights = jnp.expand_dims(jnp.zeros_like(weights), -1)
  item_weights = jnp.expand_dims(jnp.ones_like(item_indices), -1)
  interleaved_weights = utils.process_interleaved_data(
      placeholder_weights, item_weights, item_indices
  )
  # Merge item weights with class weights to get final weights.
  return jnp.multiply(
      jnp.expand_dims(weights, axis=-1),
      interleaved_weights,
  )


def _calculate_metrics(
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    prefix: str = '',
) -> dict[str, tuple[JTensor, JTensor]]:
  """Computes metrics for model predictions."""
  num_preds = jnp.sum(weights, dtype=jnp.float32)
  metrics_at_k = (1, 5, 10)
  approx_top_k = False

  # Compute metrics for each @k value
  # Save time by pre-computing top_k logits for max(k)
  approx_prefix = prefix + 'approx_' if approx_top_k else ''
  if approx_top_k:
    _, top_idx = jax.lax.approx_max_k(logits, k=max(metrics_at_k))
  else:
    _, top_idx = jax.lax.top_k(logits, k=max(metrics_at_k))

  jax.debug.print("top_idx {x}", x=jnp.squeeze(top_idx))

  # Compute recall@k
  recall_values = metrics.recall_at(
      metrics_at_k,
      logits=logits,
      cached_top_k=top_idx,
      labels=labels,
      weights=weights,
      num_preds=num_preds,
      approx=approx_top_k,
  )
  mrr_values = metrics.mrr_at(
      metrics_at_k,
      logits=logits,
      cached_top_k=top_idx,
      labels=labels,
      weights=weights,
      num_preds=num_preds,
      approx=approx_top_k,
  )
  ndcg_values = metrics.greedy_ndcg_at(
      metrics_at_k,
      logits=logits,
      cached_top_k=top_idx,
      labels=labels,
      weights=weights,
      num_preds=num_preds,
      approx=approx_top_k,
  )
  metric_values = {}
  for k, recall, mrr, ndcg in zip(
      metrics_at_k, recall_values, mrr_values, ndcg_values
  ):
    metric_values[f'{approx_prefix}recall_at_{k}'] = (
        recall,
        jnp.array(num_preds, recall.dtype),
    )
    metric_values[f'{approx_prefix}mrr_at_{k}'] = (
        mrr,
        jnp.array(num_preds, mrr.dtype),
    )
    metric_values[f'{approx_prefix}ndcg_at_{k}'] = (
        ndcg,
        jnp.array(num_preds, ndcg.dtype),
    )
  return metric_values


def compute_xent_loss_helper(
    predictions: NestedMap,
    input_batch: NestedMap,
    return_predictions: bool,
    apply_eval_sample_weights: bool = False,
    report_strict_acc: bool = False,
    vocab_size: int = 0,
    do_eval: bool = False,
) -> tuple[WeightedScalars, dict[str, Any]]:
  """Helper for computing the xent loss for Language model and Sequence model.

  Modified to compute ranking metrics for item labels.

  Args:
    predictions: A `.NestedMap` containing the keys `per_example_argmax`,
      `total_loss`, `avg_xent`, `aux_loss`, `total_weight` which corresponds to
      the output of the Softmax layer.
    input_batch: A `.NestedMap` object containing input tensors which contains
      the keys `labels` and `weights` which corresponds to the labels and the
      `weights` for each token in the sequence.
    return_predictions: Whether to return predictions, which can be more
      expensive.
    apply_eval_sample_weights: Boolean indicating whether to apply the per
      example weights from the input `eval_sample_weights` or not. When enabled,
      these per-example weights will be merged with the per token
      `input_batch.weights`.
    report_strict_acc: Whether to report strict accuracy. In general, this
      requires the entire portion of the sequence with nonzero weight be
      predicted correctly. Frequently used for eval on the Lambada dataset, in
      which case this metric is equivalent to full-word matching.
    vocab_size: Text vocab size. Used to adjust the item ids in labels.
    do_eval: Whether in eval mode. Ndcg and recall computation is only done in
      eval mode for now as it takes longer.

  Returns:
    - A dict or NestedMap containing str keys and (value, weight) pairs as
      values, where one of the entries is expected to correspond to the loss.
    - A dict containing arbitrary tensors describing something about each
      training example, where the first dimension of each tensor is the batch
      index. The base class just returns an empty dict.
  """

  if vocab_size <= 0:
    raise ValueError('vocab_size must be > 0.')
  weights = input_batch.weights
  # We are only interested in metrics for item labels.
  if 'label_item_weights' in input_batch:
    item_weights = input_batch.label_item_weights
  else:
    item_weights = jnp.reshape(
        _get_item_weights(weights, input_batch.item_indices - 1),
        jnp.shape(input_batch.labels),
    )
  # Adjust ids for item labels to match predicted labels.
  labels = jnp.where(
      item_weights > 0,
      input_batch.labels + vocab_size,
      input_batch.labels,
  )
  predicted_labels = predictions.per_example_argmax.astype(labels.dtype)
  num_preds = predictions.total_weight
  mean_acc = jnp.sum((labels == predicted_labels) * weights) / jnp.maximum(
      num_preds, 1
  )
  metric_weight = jnp.array(num_preds, predictions.avg_xent.dtype)

  if hasattr(predictions, 'avg_xent_weight'):
    avg_xent_weight = predictions.avg_xent_weight
  else:
    avg_xent_weight = metric_weight

  evaluation_metrics = NestedMap(
      total_loss=(predictions.total_loss, metric_weight),
      avg_xent=(predictions.avg_xent, avg_xent_weight),
      aux_loss=(
          predictions.aux_loss,
          jnp.array(1.0, predictions.aux_loss.dtype),
      ),
      log_pplx=(predictions.avg_xent, avg_xent_weight),
      fraction_of_correct_next_step_preds=(mean_acc, metric_weight),
      num_predictions=(num_preds, jnp.array(1.0, num_preds.dtype)),
  )
  if report_strict_acc:
    num_acc = jnp.sum(weights, axis=-1, dtype=jnp.float32)
    ## mask out padding examples
    num_acc = jax.lax.select(
        input_batch.eval_sample_weights.astype(jnp.int32),
        num_acc,
        jnp.inf * jnp.ones_like(num_acc),
    )
    num_nonpadding = jnp.sum(input_batch.eval_sample_weights)

    mean_acc_strict = jnp.sum(
        jnp.sum((labels == predicted_labels) * weights, axis=-1) == num_acc
    ) / jnp.maximum(num_nonpadding, 1)
    strict_weight = jnp.array(num_nonpadding, predictions.avg_xent.dtype)

    evaluation_metrics.acc_strict = (mean_acc_strict, strict_weight)
  if do_eval:
    example_metrics = _calculate_metrics(
        predictions.logits, labels, item_weights
    )
    evaluation_metrics.update(example_metrics)
  # The score for the sequence is the negative of the sum of per token cross
  # entropy, which is the (weighted) sum of log probs on the tokens.
  per_example_output = NestedMap(
      labels=labels, scores=-predictions.per_sequence_xent
  )
  if apply_eval_sample_weights and hasattr(input_batch, 'eval_sample_weights'):
    per_example_output.eval_sample_weights = input_batch.eval_sample_weights
  if return_predictions:
    per_example_output = predictions
  return evaluation_metrics, per_example_output


class EmbeddingInterleavedLanguageModel(models.LanguageModel):
  """Language Model with item embedding inputs interleaved with token inputs.

  Attributes:
    lm_tpl: Transformer language model layer.
  """

  lm_tpl: LayerTpl = template_field(
      interleaved_transformer_lm.InterleavedTransformerLm
  )

  def compute_loss(  # pytype: disable=signature-mismatch  # jax-ndarray
      self, predictions: NestedMap, input_batch: NestedMap
  ) -> tuple[Union[WeightedScalars, base_model.Metrics], dict[str, Any]]:
    """Computes the loss and other metrics for the given predictions.

    Args:
      predictions: The output of `compute_predictions`.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      - A dict or NestedMap containing str keys and (value, weight) pairs as
        values, where one of the entries is expected to corresponds to the loss.
      - A dict containing arbitrary tensors describing something about each
        training example, where the first dimension of each tensor is the batch
        index.
    """
    return compute_xent_loss_helper(
        predictions,
        input_batch,
        self.return_predictions,
        self.apply_eval_sample_weights,
        self.report_strict_acc,
        self.lm_tpl.vocab_size,
        self.do_eval,
    )

  def _prepare_predict_data(self, input_batch: NestedMap) -> NestedMap:
    predict_data = super()._prepare_predict_data(input_batch)

    extra_input_kwargs = predict_data.extra_input_kwargs
    # Merge item weights with class weights to get final weights.
    if (
        'label_item_weights' in input_batch
        and 'input_item_weights' in input_batch
    ):
      extra_input_kwargs['label_item_weights'] = input_batch.label_item_weights
      extra_input_kwargs['input_item_weights'] = input_batch.input_item_weights
      return predict_data

    if 'item_indices' not in input_batch:
      raise ValueError(
          '"item_indices" must present when "item_embeddings" is present.'
      )
    label_item_weights = _get_item_weights(
        predict_data.labels.class_weights, input_batch.item_indices - 1
    )
    input_item_weights = _get_item_weights(
        jnp.ones_like(predict_data.labels.class_weights),
        input_batch.item_indices,
    )
    extra_input_kwargs['label_item_weights'] = label_item_weights
    extra_input_kwargs['input_item_weights'] = input_item_weights
    return predict_data

  def _prepare_decode_data(
      self, input_batch: models.NestedMap, decoder_params: models.DecoderHParams
  ) -> models.NestedMap:
    decode_data = super()._prepare_decode_data(input_batch, decoder_params)
    input_item_weights = _get_item_weights(
        jnp.ones_like(decode_data.inputs.ids),
        input_batch.item_indices,
    )
    extra_input_kwargs = decode_data.extra_input_kwargs
    extra_input_kwargs['input_item_weights'] = input_item_weights
    return decode_data
