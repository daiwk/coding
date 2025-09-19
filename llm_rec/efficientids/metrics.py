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

"""Metric utility functions."""

from collections.abc import Sequence
import functools

import jax
from jax import numpy as jnp
from praxis import base_layer


JTensor = base_layer.JTensor


@functools.partial(jax.jit, static_argnames=['k', 'approx'])
def recall_at(
    k,
    logits,
    labels,
    weights,
    num_preds,
    approx = True,
    cached_top_k = None,
):
  """Compute recall@k from logits and labels.

  An example contains a sequence of items, some of which are masked, i.e.,
  intended for prediction. We only measure recall@k for the masked items by
  using `weights`, which represents a bitmask of masked items in the sequence.
  For each masked item in the batch: given a retrieval set of the top k scoring
  logits for that item (in `logits`), assign a score of 1.0 if any of those
  predicted items match the corresponding label, and 0 otherwise. Returns the
  average of that score across all masked items in the batch.

  Args:
    k: the set of desired ranked prediction cutoffs to match against the labels.
      Must contain k's of 1 or greater.
    logits: a [batch, sequence_length, vocab size](float32) tensor of prediction
      logits. Note that logits contains predictions for all items, not just the
      masked ones.
    labels: a [batch, sequence_length](int32) tensor of labels, where the values
      are indices into the vocabulary.
    weights: a [batch, sequence_length](int32) tensor, where 1 indicates that
      the item is a mask; 0 otherwise.
    num_preds: a scalar, the number of predictions (aka masks) present in the
      batch, used as a denominator to average recall@k. In principle, we could
      compute num_preds directly from weights, but for consistency with other
      metrics, we use the value that an upstream process computes, as is done
      with other metrics.
    approx: if True, uses jax.lax.approx_max_k instead of jax.lax.top_k to
      improve performance.
    cached_top_k: an optional pre-computed [batch, sequence_length, max(k)]
      (int32) tensor of top_k logit indices, which, if provided, will be used
      instead of re-computing the `top_k` of logits to improve performance.

  Returns:
    A List of JTensor scalars representing the average recall@k for the batch,
    as described above, for each k value provided.

  Raises:
    ValueError if k < 1.
  """
  if isinstance(k, int):
    k = [k]
  if not k or min(k) < 1:
    raise ValueError(f'{k=} must not be empty or contain values less than 1')

  # Returns the top k indices based on logit values. Note that indices are
  # comparable to label values, both of which represent an index into the
  # vocabulary.
  # Output: [batch, sequence_length, k]
  if cached_top_k is None:
    if approx:
      _, top_idx = jax.lax.approx_max_k(logits, k=max(k))
    else:
      _, top_idx = jax.lax.top_k(logits, k=max(k))
  else:
    top_idx = cached_top_k
  # Broadcast the labels to [batch, sequence_length, k] to make k copies of each
  # label. Then, compare element-wise with top_idx to see if any of the top-k
  # indices match the label.
  # Output: [batch, sequence_length, k](bool)
  vals = []
  label_hitmask = jnp.equal(top_idx, jnp.expand_dims(labels, axis=-1))

  for cutoff in k:
    label_cutoff_hitmax = label_hitmask[Ellipsis, :cutoff]

    # Determine if there were any matches at each position by reducing along the
    # k dimension.
    # Output: [batch, sequence_length](int32)
    item_hit = jnp.any(label_cutoff_hitmax, axis=-1).astype(jnp.int32)
    # Only count matches at masked positions by zeroing out non-mask matches.
    # Output: [batch, sequence_length](int32)
    real_hits_on_masked = item_hit * weights

    # Return average recall at each k value.
    vals.append(jnp.sum(real_hits_on_masked) / jnp.maximum(num_preds, 1))
  return vals


@functools.partial(jax.jit, static_argnames=['k', 'approx', 'ndcg'])
def mrr_at(
    k,
    logits,
    labels,
    weights,
    num_preds,
    approx = True,
    cached_top_k = None,
    ndcg = False,
):
  """Compute MRR@k from logits and labels.

  An example contains a sequence of items, some of which are masked, i.e.,
  intended for prediction. We only measure MRR@k for the masked items by
  using `weights`, which represents a bitmask of masked items in the sequence.
  For each masked item in the batch: given a retrieval set of the top k scoring
  logits for that item (in `logits`), assign an MRR score if any of those
  predicted items match the corresponding label, and 0 otherwise. Returns the
  average of the MRR score across all masked items in the batch.

  Args:
    k: the set of desired ranked prediction cutoffs to match against the labels.
      Must contain k's of 1 or greater.
    logits: a [batch, sequence_length, vocab size](float32) tensor of prediction
      logits. Note that logits contains predictions for all items, not just the
      masked ones.
    labels: a [batch, sequence_length](int32) tensor of labels, where the values
      are indices into the vocabulary.
    weights: a [batch, sequence_length](int32) tensor, where 1 indicates that
      the item is a mask; 0 otherwise.
    num_preds: a scalar, the number of predictions (aka masks) present in the
      batch, used as a denominator to average recall@k. In principle, we could
      compute num_preds directly from weights, but for consistency with other
      metrics, we use the value that an upstream process computes, as is done
      with other metrics.
    approx: if True, uses jax.lax.approx_max_k instead of jax.lax.top_k to
      improve performance.
    cached_top_k: If provided, uses pre-computed value as top_k logit index
      instead of computing `top_k` of logits to improve performance
    ndcg: If True, use nDCG denonmiator (log_2(rel_i + 2)) instead of mrr (rel_i
      + 1) denonimator. See `greedy_ndcg_at` for more details

  Returns:
    A List of JTensor scalars containing the average MRR@k for the batch,
    as described above, for each k value provided.

  Raises:
    ValueError if k < 1.
  """
  if isinstance(k, int):
    k = [k]
  if not k or min(k) < 1:
    raise ValueError(f'{k=} must not be empty or contain values less than 1')
  # Returns the top k indices based on logit values. Note that indices are
  # comparable to label values, both of which represent an index into the
  # vocabulary.
  # Output: [batch, sequence_length, k]

  if cached_top_k is None:
    if approx:
      _, top_idx = jax.lax.approx_max_k(logits, k=max(k))
    else:
      _, top_idx = jax.lax.top_k(logits, k=max(k))
  else:
    top_idx = cached_top_k

  # Broadcast the labels to [batch, sequence_length, k] to make k copies of each
  # label. Then, compare element-wise with top_idx to see if any of the top-k
  # indices match the label.
  # Output: [batch, sequence_length, k](bool)
  label_hitmask = jnp.equal(top_idx, jnp.expand_dims(labels, axis=-1))

  # Some seq positions don't have hits; note the seq positions with hits.
  # Output: [batch, sequence_length](bool)
  has_hits = jnp.any(label_hitmask, axis=-1)

  # Find first index with hit
  # Output: [batch, sequence_length](int32)
  hit_indices = jnp.argmax(label_hitmask, axis=-1)

  # Compute MRR for all first hits
  # Output: [batch, sequence_length](float)
  if ndcg:
    maybe_metric = 1 / jnp.log2(hit_indices + 2)
  else:
    maybe_metric = 1 / (hit_indices + 1)

  # Gate MRR at each seq position based on whether there was a hit
  # Output: [batch, sequence_length](float)
  metric = jnp.where(has_hits, maybe_metric, 0)
  assert labels.shape == metric.shape

  # Only keep the MRR scores at masked positions by zeroing out non-mask
  # matches.
  # Output: [batch, sequence_length](float)
  real_metric_on_masked = metric * weights
  assert weights.shape == real_metric_on_masked.shape

  # Return average MRR at each k value.
  return [
      jnp.sum(jnp.where(hit_indices < cutoff, real_metric_on_masked, 0))
      / jnp.maximum(num_preds, 1)
      for cutoff in k
  ]


@functools.partial(jax.jit, static_argnames=['k', 'approx'])
def greedy_ndcg_at(
    k,
    logits,
    labels,
    weights,
    num_preds,
    approx = True,
    cached_top_k = None,
):
  """Compute greedy nDCG@k from logits and labels.

  Usually, nDCG is used on labels consisting of a ranked list of ground truth,
  given as DCG/iDCG. Since we only have one valid item as a label, the ideal
  DCG@k (iDCG) will always be 1 and DCG@k will only contain a single value.
  Knowing this, we avoid calculating iDCG and calculate DCG using the rank
  of the first (and only) relevant retrieved item. In summary, under our
  usecase, nDCG is just spicy MRR with "log_2(rel_i + 2)" as the
  denominator instead of "rel_i + 1".

  An example contains a sequence of items, some masked, i.e.,
  intended for prediction. We only measure nDCG@k for the masked items
  using `weights`, representing a bitmask of masked items in the sequence.
  For each masked item in the batch: given a retrieval set of the top k scoring
  logits for that item (in `logits`), assign an nDCG score if any of those
  predicted items match the corresponding label and 0 otherwise. Returns the
  average of the nDCG score across all masked items in the batch.

  Args:
    k: the set of desired ranked prediction cutoffs to match against the labels.
      Must contain k's of 1 or greater.
    logits: a [batch, sequence_length, vocab size](float32) tensor of prediction
      logits. Note that logits contains predictions for all items, not just the
      masked ones.
    labels: a [batch, sequence_length](int32) tensor of labels, where the values
      are indices into the vocabulary.
    weights: a [batch, sequence_length](int32) tensor, where 1 indicates that
      the item is a mask; 0 otherwise.
    num_preds: a scalar, the number of predictions (aka masks) present in the
      batch, used as a denominator to average recall@k. In principle, we could
      compute num_preds directly from weights, but for consistency with other
      metrics, we use the value that an upstream process computes, as is done
      with other metrics.
    approx: if True, uses jax.lax.approx_max_k instead of jax.lax.top_k to
      improve performance.
    cached_top_k: If provided, uses pre-computed value as top_k logit index
      instead of computing `top_k` of logits to improve performance

  Returns:
    A List of JTensor scalars representing the average nDCG@k for the batch,
    as described above, for each k value provided.

  Raises:
    ValueError if k < 1.
  """
  return mrr_at(
      k, logits, labels, weights, num_preds, approx, cached_top_k, ndcg=True
  )