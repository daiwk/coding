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

from collections.abc import Mapping, Optional

import seqio
import tensorflow.compat.v2 as tf

autoregressive_inputs = seqio.utils.make_autoregressive_inputs

class ItemDecoderTfExampleFeatureConverterPack(seqio.FeatureConverter):
  """Tf examples feature converter for Interleaved language and item decoder.

  Performs padding for different features. The padded values are different for
  different features.
  """

  FEATURES = (
      "ids",
      "inputs_indicator",
      "prefix_lengths",
      "paddings",
      "labels",
      "weights",
      "label_item_weights",
      "input_item_weights",
  )

  TASK_FEATURES = {
      "ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "inputs_indicator": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "prefix_lengths": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "paddings": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "labels": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "label_item_weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "input_item_weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }

  MODEL_FEATURES = {
      "ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "inputs_indicator": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "prefix_lengths": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "paddings": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "labels": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "label_item_weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "input_item_weights": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "ids_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "inputs_indicator_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "prefix_lengths_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "paddings_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "labels_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "weights_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "label_item_weights_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "input_item_weights_segment_ids": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
  }

  TEMP_UNSEEN_ID: float = 2

  def __init__(
      self,
      pack = True,
      use_custom_packing_ops = True,
      apply_length_check = True,
      bos_id = 0,
      passthrough_features = None,
  ):
    super().__init__(
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id,
        passthrough_features=passthrough_features,
    )

  # Packing code treats '0' as special by skipping it. To get the right behavior
  # the weight / inputs_indicator 0 is mapped to TEMP_UNSEEN_ID and after
  # packing it is mapped back to 0.
  def _zero_to_temp_value(
      self, features
  ):
    # Mapping does not support __setitem__, hence a shallow copy to a dict.
    features = dict(features)
    features["weights"] = tf.where(
        features["weights"] == 0, self.TEMP_UNSEEN_ID, features["weights"]
    )
    features["label_item_weights"] = tf.where(
        features["label_item_weights"] == 0, self.TEMP_UNSEEN_ID, features["label_item_weights"]
    )
    features["input_item_weights"] = tf.where(
        features["input_item_weights"] == 0, self.TEMP_UNSEEN_ID, features["input_item_weights"]
    )
    features["inputs_indicator"] = tf.where(
        features["inputs_indicator"] == 0,
        int(self.TEMP_UNSEEN_ID),
        features["inputs_indicator"],
    )
    return features

  def _temp_value_to_zero(
      self, features
  ):
    # Mapping does not support __setitem__, hence a shallow copy to a dict.
    features = dict(features)
    features["weights"] = tf.where(
        features["weights"] == self.TEMP_UNSEEN_ID, 0, features["weights"]
    )
    features["label_item_weights"] = tf.where(
        features["label_item_weights"] == self.TEMP_UNSEEN_ID, 0, features["label_item_weights"]
    )
    features["input_item_weights"] = tf.where(
        features["input_item_weights"] == self.TEMP_UNSEEN_ID, 0, features["input_item_weights"]
    )
    features["inputs_indicator"] = tf.where(
        features["inputs_indicator"] == int(self.TEMP_UNSEEN_ID),
        0,
        features["inputs_indicator"],
    )
    return features

  def _drop_unused_features(
      self, features
  ):
    out = {
        "ids": features["ids"],
        "labels": features["labels"],
        "weights": features["weights"],
        "prefix_lengths": features["prefix_lengths"],
        "label_item_weights": features["label_item_weights"],
        "input_item_weights": features["input_item_weights"],
        "inputs_indicator": features["inputs_indicator"],
        "paddings": features["paddings"],
    }
    if self.pack:
      out["segment_ids"] = features["ids_segment_ids"]
      out["segment_pos"] = features["ids_positions"]

    # Fix the paddings (after pad or packing there might be extra 0's).
    # To correctly compute the padding the 'labels' != 0 is the ground truth.
    out["paddings"] = 1 - seqio.feature_converters.non_padding_position(
        out["labels"]
    )

    return out

  def _convert_features(
      self, ds, task_feature_lengths
  ):
    """ItemDecoderTfExampleFeatureConverter does not have this method."""
    raise Exception(
        'ItemDecoderTfExampleFeatureConverter does not have this method.'
    )

  def _maybe_fix_feature_lengths(
      self, task_feature_lengths
  ):
    task_feature_lengths = dict(task_feature_lengths)
    expected_keys = set(self.FEATURES)
    actual_keys = set(task_feature_lengths.keys())
    if expected_keys == actual_keys:
      return task_feature_lengths
    len_inputs = task_feature_lengths["inputs"]
    common_length = len_inputs
    if "targets" in task_feature_lengths:
      len_targets = task_feature_lengths["targets"]
      common_length += len_targets
    if "suffixes" in task_feature_lengths:
      len_suffix = task_feature_lengths["suffixes"]
      common_length += len_suffix
    return {k: common_length for k in expected_keys}

  def get_model_feature_lengths(
      self, task_feature_lengths
  ):
    """Define the length relationship between input and output features."""
    task_feature_lengths = self._maybe_fix_feature_lengths(task_feature_lengths)
    return task_feature_lengths

  def __call__(
      self, ds, task_feature_lengths
  ):
    # The task feature lengths must match the keys.
    task_feature_lengths = self._maybe_fix_feature_lengths(task_feature_lengths)
    if self.pack:
      ds = ds.map(self._zero_to_temp_value, num_parallel_calls=tf.data.AUTOTUNE)
    ds = self._pack_or_pad(ds, task_feature_lengths)
    ds = ds.map(self._drop_unused_features, num_parallel_calls=tf.data.AUTOTUNE)

    # Revert self.TEMP_UNSEED_ID in the case of packing.
    if self.pack:
      ds = ds.map(self._temp_value_to_zero, num_parallel_calls=tf.data.AUTOTUNE)

    return ds