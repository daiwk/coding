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

"""Utils."""

import jax
import jax.numpy as jnp
import seqio
import tensorflow as tf
from praxis import base_input
from praxis import py_utils
from praxis import pytypes


class ILMInputProcessor:
  """ILM input TF processor.

  Example usage:
  ```
  processor = ILMInputProcessor(query_length=2, vocab=ULM_VOCAB_V0)
  processor.add_sos()
  processor.add_text('a')
  processor.add_item(item_embedding1)
  processor.add_text('c')
  processor.add_item(item_embedding2)
  processor.add_text('d')
  processor.add_eos()

  inputs = processor.get_inputs()
  ```
  - `inputs['ids']` are item-text interleaved token ids, `<s> a <ctrl0> <ctrl0>
    c <plh> <plh> d </s>`, where the item token ids (`<plh>`) are placeholders.
  - `inputs['item_embeddings']` is a concatenation of all flattened item
    embeddings.
  - `inputs['item_indices']` will map item embeddings to placeholder token
    positions, i.e. the positions of `<plh>`.
  """

  def __init__(
      self,
      query_length,
      vocab,
      sos_id = 0,
      eos_id = 1,
      plh_id = 102,
  ):
    """Initializes ILMInputProcessor.

    Args:
      query_length: Number of queries (tokens) per item. If zero, the
        concatenated sequence will be text-only.
      vocab: The SPM vocabulary.
      sos_id: Start of sentence id.
      eos_id: End of sentence id.
      plh_id: The item placeholder id.
    """
    self.query_length = query_length
    self.vocab = vocab
    self.sos_id = sos_id
    self.eos_id = eos_id
    self.plh_id = plh_id
    self.ids = tf.constant([], dtype=tf.int32)
    self.labels = tf.constant([], dtype=tf.int32)
    self.inputs_indicator = tf.constant([], dtype=tf.int32)
    self.item_indices = tf.constant([], dtype=tf.int32)
    self.label_item_weights = tf.constant([], dtype=tf.int32)
    self.input_item_weights = tf.constant([], dtype=tf.int32)

  def add_text(self, text, is_prefix = True):
    """Adds text segment to inputs.

    Args:
      text: The text segment to add to inputs.
      is_prefix: Whether the text segment is in prefix. If true, bi-directional
        attention is applied for this text segment, if False, causal attention
        is applied for this text segment.
    """
    if isinstance(text, str):
      text = tf.constant(text)
    text_ids = self.vocab.encode_tf(text)
    self.ids = tf.concat([self.ids, text_ids], axis=0)
    self.labels = tf.concat([self.labels, text_ids], axis=0)
    if is_prefix:
      indicator = tf.ones_like(text_ids)
    else:
      indicator = tf.zeros_like(text_ids)
    self.inputs_indicator = tf.concat(
        [self.inputs_indicator, indicator], axis=0
    )
    self.input_item_weights = tf.concat(
        [self.input_item_weights, tf.zeros_like(text_ids)], axis=0
    )
    self.label_item_weights = tf.concat(
        [self.label_item_weights, tf.zeros_like(text_ids)], axis=0
    )

  def add_item(
      self,
      item_embeddings,
      is_prefix = False,
      item_id = None,
  ):
    """Adds item embedding and placeholder tokens to inputs.

    Args:
      item_embeddings: The item embedding to add to inputs.
      is_prefix: Whether the item is a label. If true, the input indicator will
        be set to 0, otherwise 1.
      item_id: The item id to add to ids. If None, the placeholder id will be
        used.
    """
    start = tf.shape(self.ids)[0]
    self.item_indices = tf.concat(
        [self.item_indices, tf.range(start, self.query_length + start)], axis=0
    )
    if item_id is not None:
      if is_prefix:
        self.ids = tf.concat(
            [self.ids, tf.fill([self.query_length], self.plh_id)], axis=0
        )
      else:
        self.ids = tf.concat(
            [self.ids, tf.fill([self.query_length], item_id)], axis=0
        )
      self.labels = tf.concat(
          [self.labels, tf.fill([self.query_length], item_id)], axis=0
      )
      self.input_item_weights = tf.concat(
          [self.input_item_weights, tf.fill([self.query_length], 1)], axis=0
      )
    else:
      self.ids = tf.concat(
          [self.ids, tf.fill([self.query_length], self.plh_id)], axis=0
      )
      self.labels = tf.concat(
          [self.labels, tf.fill([self.query_length], self.plh_id)], axis=0
      )
      self.input_item_weights = tf.concat(
          [self.input_item_weights, tf.fill([self.query_length], 0)], axis=0
      )
    self.inputs_indicator = tf.concat(
        [
            self.inputs_indicator,
            tf.fill([self.query_length], 0 if is_prefix else 1),
        ],
        axis=0,
    )
    self.label_item_weights = tf.concat(
        [self.label_item_weights, tf.fill([self.query_length], 1 if is_prefix else 0)], axis=0
    )

  def add_sos(self):
    """Adds sos token to inputs."""
    self.ids = tf.concat([self.ids, tf.constant([self.sos_id])], axis=0)
    self.labels = tf.concat([self.labels, tf.constant([self.sos_id])], axis=0)
    self.inputs_indicator = tf.concat(
        [self.inputs_indicator, tf.constant([1])], axis=0
    )
    self.input_item_weights = tf.concat(
        [self.input_item_weights, tf.constant([0])], axis=0
    )
    self.label_item_weights = tf.concat(
        [self.label_item_weights, tf.constant([0])], axis=0
    )

  def add_eos(self):
    """Adds eos token to inputs."""
    self.ids = tf.concat([self.ids, tf.constant([self.eos_id])], axis=0)
    self.labels = tf.concat([self.labels, tf.constant([self.eos_id])], axis=0)
    self.inputs_indicator = tf.concat(
        [self.inputs_indicator, tf.constant([0])], axis=0
    )
    self.input_item_weights = tf.concat(
        [self.input_item_weights, tf.constant([0])], axis=0
    )
    self.label_item_weights = tf.concat(
        [self.label_item_weights, tf.constant([0])], axis=0
    )

  def get_inputs(self):
    """Returns inputs for the ILM model."""
    return {
        'ids': self.ids,
        'inputs_indicator': self.inputs_indicator,
        'prefix_lengths': tf.reduce_sum(self.inputs_indicator)[tf.newaxis],
        'paddings': tf.zeros_like(self.ids, dtype=tf.float32),
        'item_indices': self.item_indices,
        'labels': tf.pad(self.labels[1:], [[0, 1]]),
        'weights': tf.cast(
            tf.pad(1 - self.inputs_indicator[1:], [[0, 1]]), tf.float32
        ),
        'input_item_weights': self.input_item_weights,
        'label_item_weights': tf.pad(self.label_item_weights[1:], [[0, 1]]),
    }

  @classmethod
  def pad(
      cls, inputs, feature_lengths
  ):
    """Returns padded or truncated inputs."""
    # Feature name to padding (max_len, constant_values).
    for name, max_len in feature_lengths.items():
      if name == 'paddings':
        constant_values = 1.0
      elif name == 'item_indices':
        constant_values = feature_lengths['ids']
      else:
        constant_values = 0
      inputs[name] = tf.pad(
          inputs[name], [[0, max_len]], constant_values=constant_values
      )[:max_len]
    return inputs


def process_interleaved_data(
    input_embeddings,
    item_embeddings,
    item_indices,
):
  """Returns interleaved item-text embeddings.

  The input embeddings should already contain placeholders for item embeddings.

  Args:
    input_embeddings: <float>[B, L, D], input embeddings with item token
      placeholders.
    item_embeddings: <float>[B, M, D], concatenated item embeddings.
    item_indices: <int>[B, M], the indices of item tokens in input tokens.

  Returns:
    <float>[B, L, D], input embeddings with item token placeholders replaced by
    embeddings.
  """
  batch_size = input_embeddings.shape[0]
  embed_size = input_embeddings.shape[-1]
  num_item_tokens = item_indices.shape[1]

  # [B * M]
  inds = jnp.stack(
      [
          jnp.arange(batch_size).repeat(num_item_tokens),
          item_indices.reshape(-1),
      ],
      axis=1,
  )
  # [B * M, D]
  item_embeddings = item_embeddings.astype(input_embeddings.dtype)
  item_embeddings = item_embeddings.reshape(-1, embed_size)
  sdm = jax.lax.ScatterDimensionNumbers(
      update_window_dims=(1,),
      inserted_window_dims=(0, 1),
      scatter_dims_to_operand_dims=(0, 1),
  )
  # [B, L, D]
  input_embeddings = jax.lax.scatter(
      input_embeddings, inds, item_embeddings, sdm, mode='drop'
  )
  return input_embeddings

class ItemDecoderInputSpecsProviderPack(base_input.BaseInputSpecsProvider):
  """Item Decoder Model input specs provider.

  Attributes:
    per_core_batch_size: Per-core batch size.
    num_items: Number of items per example.
    item_embedding_size: The size of each embedding.
    input_max_len: Max number of tokens in the inputs.
  """

  per_core_batch_size: int | None = None
  num_items: int = 0
  input_max_len: int = 0

  def get_input_specs(self):
    """Returns specs from the input pipeline for model init."""
    if self.per_core_batch_size is None:
      raise ValueError('per_core_batch_size is not set.')
    bs, _ = batch_size
    old = py_utils.NestedMap(
        ids=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        inputs_indicator=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        paddings=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        labels=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        weights=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        label_item_weights=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        input_item_weights=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len),
            dtype=jnp.int32,
        ),
        segment_ids=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len), dtype=jnp.int32
        ),
        segment_pos=jax.ShapeDtypeStruct(
            shape=(bs, self.input_max_len), dtype=jnp.int32
        ),
        eval_sample_weights=jax.ShapeDtypeStruct(
            shape=[bs], dtype=jnp.float32
        ),
        prefix_lengths=jax.ShapeDtypeStruct(
            shape=[bs, self.input_max_len], dtype=jnp.int32
        ),
    )

    new = py_utils.NestedMap.FromNestedDict({
        "_seqio_provenance/index_within_shard": jax.ShapeDtypeStruct(
            shape=[bs], dtype=jnp.int64
        ),
        "_seqio_provenance/num_shards": jax.ShapeDtypeStruct(
            shape=[bs], dtype=jnp.int32
        ),
        "_seqio_provenance/shard_index": jax.ShapeDtypeStruct(
            shape=[bs], dtype=jnp.int32
        ),
    })
    old.update(new)
    return old
