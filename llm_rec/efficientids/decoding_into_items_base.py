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

"""base for experiments with items."""

import pickle
from typing import cast

from absl import logging
import fiddle as fdl
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import tasks_lib
from praxis import base_input
from praxis import base_layer
from praxis import decoder_hparams
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis.layers import activations
from praxis.layers import linears
from praxis import schedules
from praxis.layers import transformer_models
import seqio
import tensorflow.compat.v2 as tf

from paxml import seqio_input
import utils
import embedding_adapter_prompt as eap_lib
import feature_converters as feature_converter_lib
import interleaved_transformer_lm as item_lm


@experiment_registry.register()
class ItemDecoderPALMBase(base_experiment.BaseExperiment):
  """Item decoder finetuning based on PALM checkpoints."""

  ITEM_EMBEDDING_SIZE: int = 128  # WALS item embedding size.
  ITEM_VOCAB_SIZE: int = 17388
  ITEM_EMBEDDING_CKPT: str = ''
  CLUSTER_EMBEDDING_CKPT: str = ''
  CLUSTERING_PKL: str = ''

  TRAINABLE_CLUSTER_EMBEDDINGS: bool = False
  TRAINABLE_ITEM_EMBEDDINGS: bool = False
  FULL_SOFTMAX: bool = False
  NUM_ITEM_CLUSTERS: int = 100

  ITEM_DNN_MLP_HIDDEN_DIMS: int = 128
  ITEM_DNN_MLP_NUM_LAYERS: int = 1

  CHECKPOINT_SAFE_LOAD = True
  CHECKPOINT_LOAD_STEP = False

  # Set lm.
  LM_TYPE = PALM1B
  LM_CKPT_PATH: str | None = ''
  LM_CKPT_STEP: int = 0
  LM_OVERRIDE_RULES: list[tuple[str, str]] = [
      (r'params/(.*)$', 'params/{}'),
  ]
  LM_IGNORE_RULES: list[str] = [
      r'params/lm/softmax/item_input_dnn/.*',
      r'params/lm/softmax/item_output_dnn/.*',
      r'params/lm/softmax/item_b.*',
      r'params/lm/softmax/logits_ffn_item/.*',
      r'params/lm/softmax/logits_ffn_cluster/.*',
  ]

  # Set freeze hparams
  BPROP_VARIABLE_EXCLUSION: list[str] = [
      # Freeze embedding tables.
  ]

  """ BPROP_VARIABLE_EXCLUSION: list[str] = [
      # Freeze everything except adapters.
      r'params/lm/position_emb/.*',
      r'params/lm/embedding_lookup/.*',
      r'params/lm/transformer/.*',
      r'params/lm/final_ln/.*',
      r'params/lm/softmax/logits_ffn/.*',
      r'params/lm/softmax/logits_ffn_item/.*',
  ] """

  """ BPROP_VARIABLE_EXCLUSION: list[str] = [
      # Freeze LM.
      r'params/lm/position_emb/.*',
      r'params/lm/embedding_lookup/.*',
      r'params/lm/transformer/.*',
      r'params/lm/final_ln/.*',
      r'params/lm/softmax/logits_ffn/.*',
  ] """

  # Set input hparams.
  PREPEND_SOS: bool = True
  SOS_ID: int = 0
  EOS_ID: int = 1
  PLH_ID: int = 102  # <ctrl89> in PALM vocab.
  MAX_NUM_ITEMS: int = 50
  MAX_INPUT_LEN: int = 100

  # Train data params
  TASK_NAME: str = 'id_video_games'
  PERCORE_BATCH_SIZE: int = 16
  DETERMINISTIC_INPUT: bool = False
  REMOVE_PROVENANCE_FC: bool = True

  # Eval data params
  EVAL_TASK_NAMES: list[str] = ['id_video_games']
  EVAL_SPLIT_NAMES: list[str] = ['test']
  PERCORE_EVAL_BATCH_SIZE: int = 16
  # How many batches to include during eval. If None, eval on the entire dataset
  NUM_EVAL_BATCHES: int | None = None
  EVAL_DETERMINISTIC_INPUT: bool = False

  # Decode data params
  DECODE_TASK_NAME: str = 'id_video_games'
  PERCORE_DECODE_BATCH_SIZE: int = 16
  DECODE_DETERMINISTIC_INPUT: bool = False

  FPROP_DTYPE: jnp.dtype = jnp.float32
  MODEL_DTYPE: jnp.dtype = jnp.float32

  TRAINING_OPTIMIZED_SHARDING: bool = True

  # Optimizer configs
  WARMUP_STEPS: int = 10_000
  MAX_STEPS: int = 100_000
  LEARNING_RATE: float = 1e-5
  WEIGHT_DECAY: float = 1e-2
  LEARNING_RATE_DECAY_END: int = 100_000
  LOSS_NAME: str = 'total_loss'
  INPUT_DROPOUT_PROB: float = 0.0
  DROPOUT_PROB: float = 0.5
  LABEL_SMOOTHING: float = 0.0
  SECOND_LABEL_SMOOTHING: float = 0.0
  CLIP_GRADIENT_NORM_TO_VALUE: float | None = 1.0
  LAYERWISE_ADAPTATION: bool = False
  SOFT_CAP_LOGITS: float | None = 0.0

  # Set checkpointing hparams.
  SUMMARY_INTERVAL_STEPS: int = 1_000
  EVAL_INTERVAL_STEPS: int = 1_00
  SAVE_INTERVAL_STEPS: int = 10_000
  DECODE_INTERVAL_STEPS: int = 10_000

  # Set decode hparams.
  DECODING_MODE: transformer_models.LanguageModelType = (
      transformer_models.LanguageModelType.PREFIX
  )
  DECODE_ALGORITHM: str = 'greedy'
  FPROP_FOR_PREFIX: bool = True
  MAX_DECODE_STEPS: int = 32
  DECODE_SEQ_LEN: int = MAX_INPUT_LEN + MAX_DECODE_STEPS

  ENFORCE_INPUT_SPECS: bool = True
  ENABLE_SEQUENCE_PACKING = False
  CORRECTION: bool = False

  ICI_MESH_SHAPE: list[int] = [1, 64, 1]
  MESH_AXIS_NAMES: list[str] = ['replica', 'data', 'mdl']

  def _get_task_feature_lengths(self) -> dict[str, int]:
    if self.ENABLE_SEQUENCE_PACKING:
      return {
          'ids': self.MAX_INPUT_LEN,
          'inputs_indicator': self.MAX_INPUT_LEN,
          'prefix_lengths': self.MAX_INPUT_LEN,
          'paddings': self.MAX_INPUT_LEN,
          #'item_embeddings': self.MAX_NUM_ITEMS * self.ITEM_EMBEDDING_SIZE,
          #'item_indices': self.MAX_NUM_ITEMS,
          'labels': self.MAX_INPUT_LEN,
          'weights': self.MAX_INPUT_LEN,
          'input_item_weights': self.MAX_INPUT_LEN,
          'label_item_weights': self.MAX_INPUT_LEN,
      }
    else:
      return {
          'ids': self.MAX_INPUT_LEN,
          'inputs_indicator': self.MAX_INPUT_LEN,
          'prefix_lengths': 1,
          'paddings': self.MAX_INPUT_LEN,
          'item_indices': self.MAX_NUM_ITEMS,
          'labels': self.MAX_INPUT_LEN,
          'weights': self.MAX_INPUT_LEN,
      }

  def _get_feature_converter(self) -> seqio.FeatureConverter:
    if self.ENABLE_SEQUENCE_PACKING:
      return seqio_input.RemoveProvenance()
    else:
      return feature_converter_lib.ItemDecoderTfExampleFeatureConverter()

  def _add_dataset_hparams(
      self,
      mixture_name: str,
      percore_batch_size: int,
      deterministic_input: bool = False,
      remove_provenance_fc: bool = True,
      num_batches_to_skip: int | None = None,
      training: bool = True,
      split_name: str = 'test',
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    """Create HParams from mixture as train dataset."""
    batch_size, num_infeed_hosts = #batchsize and num_infeed_hosts
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        mixture_name=mixture_name,
        batch_size=batch_size,
        num_infeed_hosts=num_infeed_hosts,
        split_name=split_name,
        task_feature_lengths=self._get_task_feature_lengths(),
        feature_converter=self._get_feature_converter(),
        is_training=training,
        input_random_seed=None,
        num_batches_to_skip=num_batches_to_skip,
    )
    if not training:
      logging.info(
          'Setting num batches for eval: %s',
          p.mixture_name,
      )
      p.eval_loop_num_batches = self.NUM_EVAL_BATCHES
      if not self.NUM_EVAL_BATCHES:
        logging.info(
            'eval_loop_num_batches is set to None, therefore will evaluate on'
            ' the entire eval set'
        )
        p.eval_loop_num_batches = -1
        p.repeat = False
        p.reset_for_eval = True
      else:
        logging.info(
            'eval_loop_num_batches is set to %d, therefore each eval will'
            ' progressively iterate through approxmiately %d items in the eval'
            ' set',
            self.NUM_EVAL_BATCHES,
            self.NUM_EVAL_BATCHES * batch_size * num_infeed_hosts,
        )
        p.repeat = True
        p.reset_for_eval = False
    if deterministic_input:
      p = seqio_input.configure_deterministic_input(
          p, remove_provenance_fc=remove_provenance_fc
      )

    return p

  def datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    train_dataset = self._add_dataset_hparams(
        mixture_name=self.TASK_NAME,
        percore_batch_size=self.PERCORE_BATCH_SIZE,
        deterministic_input=self.DETERMINISTIC_INPUT,
        remove_provenance_fc=self.REMOVE_PROVENANCE_FC,
        training=True,
        split_name='train'
    )
    eval_datasets = [
        self._add_dataset_hparams(
            mixture_name=self.EVAL_TASK_NAMES[0],
            percore_batch_size=self.PERCORE_EVAL_BATCH_SIZE,
            deterministic_input=self.EVAL_DETERMINISTIC_INPUT,
            remove_provenance_fc=self.REMOVE_PROVENANCE_FC,
            training=False,
            split_name=split_name,
        )
        for split_name in self.EVAL_SPLIT_NAMES
    ]
    return [train_dataset, *eval_datasets]

  def decoder_datasets(self) -> list[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns the list of dataset parameters for decoder."""
    return []

  def get_input_specs_provider_params(
      self,
  ) -> pax_fiddle.Config[base_input.BaseInputSpecsProvider]:
    """Returns the config of the input specs provider.

    Returns:
      An InputSpecsProvider instance.
    """
    if self.ENABLE_SEQUENCE_PACKING:
      return pax_fiddle.Config(
          utils.ItemDecoderInputSpecsProviderPack,
          per_core_batch_size=self.PERCORE_BATCH_SIZE,
          num_items=self.MAX_NUM_ITEMS,
          input_max_len=self.MAX_INPUT_LEN,
      )
    else:
      return pax_fiddle.Config(
          utils.ItemDecoderInputSpecsProvider,
          per_core_batch_size=self)

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='item_decoder')
    task_p.model = self._model_hparams()

    task_p = self._configure_task(task_p)

    task_p.train.enforce_input_specs = False

    return task_p

  def _optimizer(self) -> pax_fiddle.Config[optimizers.BaseOptimizer]:
    return pax_fiddle.Config(
        optimizers.ShardedAdafactor,
        decay_method='adam',
        beta1=0.9,
        decay_adam=0.999,
        weight_decay=self.WEIGHT_DECAY,
        clip_threshold=self.CLIP_GRADIENT_NORM_TO_VALUE,
        layerwise_adaptation=self.LAYERWISE_ADAPTATION,
    )

  def _def_init_from_checkpoint_rules(
      self, task_p: pax_fiddle.Config[tasks_lib.SingleTask]
  ) -> dict[str, tasks_lib.CheckpointLoadingRules]:
    rules = {}
    if self.LM_CKPT_PATH:
      rules[self.LM_CKPT_PATH] = tasks_lib.CheckpointLoadingRules(
          task_p=task_p.clone(),
          load_rules=self.LM_OVERRIDE_RULES,
          ignore_rules=self.LM_IGNORE_RULES,
          load_opt_states=False,
          load_step=self.CHECKPOINT_LOAD_STEP,
          step=self.LM_CKPT_STEP,
          safe_load=self.CHECKPOINT_SAFE_LOAD,
          input_specs_provider_p=(self.get_input_specs_provider_params()),
      )
    if self.ITEM_EMBEDDING_CKPT:
      rules[self.ITEM_EMBEDDING_CKPT] = tasks_lib.CheckpointLoadingRules(
          task_p=task_p.clone(),
          load_rules=[
              (
                  r'params/lm/softmax/logits_ffn_item/(.*)$',
                  'params/lm/softmax/logits_ffn_item/{}',
              )
          ],
          load_opt_states=False,
          load_step=self.CHECKPOINT_LOAD_STEP,
          step=100000,  # checkpoint_step
          safe_load=self.CHECKPOINT_SAFE_LOAD,
          input_specs_provider_p=self.get_input_specs_provider_params(),  # pytype: disable=attribute-error
      )
    if self.CLUSTER_EMBEDDING_CKPT and not self.FULL_SOFTMAX:
      rules[self.CLUSTER_EMBEDDING_CKPT] = tasks_lib.CheckpointLoadingRules(
          task_p=task_p.clone(),
          load_rules=[
              (
                  r'params/lm/softmax/logits_ffn_cluster/linear/(.*)$',
                  'params/lm/softmax/logits_ffn_cluster/linear/{}',
              )
          ],
          ignore_rules=[r'params/lm/softmax/logits_ffn_cluster/bias/(.*)'],
          load_opt_states=False,
          load_step=self.CHECKPOINT_LOAD_STEP,
          step=100000,  # checkpoint_step
          safe_load=self.CHECKPOINT_SAFE_LOAD,
          input_specs_provider_p=self.get_input_specs_provider_params(),  # pytype: disable=attribute-error
      )
    return rules

  def _configure_task(
      self, task_p: pax_fiddle.Config[tasks_lib.SingleTask]
  ) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Configures commonly used task_p settings."""

    # Fold prng key per each batch index. This makes the decoding outputs
    # different for the identical inputs if they appear at different batches.
    # This is desired if we want to have better diversity of decode outputs, for
    # example, for class-conditioned image generation, we want to generate
    # diverse (i.e. different) images for the same class id.
    task_p.decode.prng_key_fold_with_batch_index = True

    train_p = task_p.train

    # Set learner and optimizer.
    train_p.learner = self._learner()
    
    train_p.learner.optimizer.learning_rate = self.LEARNING_RATE
    if self.WEIGHT_DECAY is not None:
      train_p.learner.optimizer.weight_decay = self.WEIGHT_DECAY

    # Learning rate schedule
    decay_end = self.LEARNING_RATE_DECAY_END
    train_p.learner.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=self.WARMUP_STEPS,
        decay_start=self.WARMUP_STEPS + 1,
        decay_end=decay_end,
        min_ratio=0.1,
        max=1.0,
    )

    # Set sharding annotations.
    mesh_shape = self.ICI_MESH_SHAPE or self._mesh_shape()
    if mesh_shape is not None:
      model = task_p.model
      model.ici_mesh_shape = mesh_shape
      if (dcn_mesh_shape := self._dcn_mesh_shape()) is not None:
        model.dcn_mesh_shape = dcn_mesh_shape
      model.mesh_axis_names = self.MESH_AXIS_NAMES

      batch_split = self.BATCH_SPLIT_AXES
      train_p.inputs_split_mapping = py_utils.NestedMap(
          map_6d=(batch_split, None, None, None, None, None),
          map_5d=(batch_split, None, None, None, None),
          map_4d=(batch_split, None, None, None),
          map_3d=(batch_split, None, None),
          map_2d=(batch_split, None),
          map_1d=(batch_split,),
      )

    # Set model loading rules.
    train_p.init_from_checkpoint_rules = self._def_init_from_checkpoint_rules(
        task_p
    )

    # Set summary, checkpointing, evaluation and decoding.
    train_p.variable_norm_summary = True
    train_p.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    train_p.summary_accumulate_interval_steps = (
        self.SUMMARY_ACCUMULATE_INTERVAL_STEPS
    )

    train_p.num_train_steps = self.MAX_STEPS
    train_p.save_interval_steps = self.SAVE_INTERVAL_STEPS
    train_p.save_max_to_keep = self.SAVE_MAX_TO_KEEP

    train_p.eval_skip_train = True  # Disable eval of train input data.
    train_p.eval_interval_steps = self.EVAL_INTERVAL_STEPS

    train_p.decode_interval_steps = self.DECODE_INTERVAL_STEPS

    # Use ema state to eval/decode when run decoding during the training.
    if train_p.learner.optimizer.ema_decay > 0.0:
      train_p.decode_use_ema_states = True
      train_p.eval_use_ema_states = True

    if self.DEBUG_REDUCE_SUMMARY_AND_SKIP_VALID_CHECK:
      train_p.variable_norm_summary = False
      train_p.learner.grad_norm_summary = False
      train_p.learner.var_norm_summary = False
      train_p.learner.check_valid_step = False
    return task_p

  def _model_hparams(self):
    lm_p = pax_fiddle.Config(
        item_lm.InterleavedTransformerLm, name='transformer_lm'
    )
    logging.info(
        'Constructing model with %d item clusters', self.NUM_ITEM_CLUSTERS
    )

    original_lm_p = self.LM_TYPE().task().model.lm_tpl
    original_lm_p.stacked_transformer_tpl.block.input_dropout_prob = (
        self.INPUT_DROPOUT_PROB
    )
    original_lm_p.stacked_transformer_tpl.block.dropout_prob = (
        self.DROPOUT_PROB
    )
    original_lm_p.packed_input = self.ENABLE_SEQUENCE_PACKING
    # Set sharding.
    replica_axis, data_axis, mdl_axis = self.MESH_AXIS_NAMES[-3:]
    original_lm_cls = cast(
        transformer_models.TransformerLm, fdl.get_callable(original_lm_p)
    )
    original_lm_p = original_lm_cls.set_sharding_params_v1(
        original_lm_p,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        ici_mesh_shape=self.ICI_MESH_SHAPE,
        dcn_mesh_shape=None,
        mesh_axis_names=self.MESH_AXIS_NAMES,
        training_optimized=self.TRAINING_OPTIMIZED_SHARDING,
        batch_axes=None,
    )
    lm_p.copy_fields_from(original_lm_p)

    # Override lm_p according to item-language needs.
    lm_p.max_num_items = self.MAX_NUM_ITEMS
    lm_p.trainable_item_embeddings = self.TRAINABLE_ITEM_EMBEDDINGS

    logging.info('Initializing the embedding softmax')
    embedding_softmax_p = pax_fiddle.Config(item_lm.SharedEmbeddingSoftmax)
    embedding_softmax_p.copy_fields_from(lm_p.softmax_tpl)
    embedding_softmax_p.chunk_size = (
        0  # TODO: Unset this to improve HBM in future.
    )
    embedding_softmax_p.full_softmax = self.FULL_SOFTMAX
    embedding_softmax_p.num_item_classes = self.ITEM_VOCAB_SIZE
    embedding_softmax_p.num_clusters = self.NUM_ITEM_CLUSTERS
    embedding_softmax_p.trainable_cluster_embeddings = (
        self.TRAINABLE_CLUSTER_EMBEDDINGS
    )
    embedding_softmax_p.item_input_dims = self.ITEM_EMBEDDING_SIZE
    embedding_softmax_p.label_smoothing_apply_for_eval = False
    embedding_softmax_p.label_smoothing_prob = self.LABEL_SMOOTHING
    embedding_softmax_p.second_label_smoothing_prob = self.SECOND_LABEL_SMOOTHING
    embedding_softmax_p.soft_cap_logits = self.SOFT_CAP_LOGITS

    if self.CLUSTERING_PKL is not None:
      with tf.io.gfile.GFile(self.CLUSTERING_PKL, 'rb') as f:
        cluster_assignments = pickle.load(f)
        cluster_indices = pickle.load(f)
        in_cluster_id = pickle.load(f)
        cluster_means = pickle.load(f)
      embedding_softmax_p.cluster_assignments = cluster_assignments
      embedding_softmax_p.cluster_indices = cluster_indices
      embedding_softmax_p.in_cluster_id = in_cluster_id
      embedding_softmax_p.cluster_embeddings = cluster_means

    logging.info('Initializing the item feed forward')
    item_fft = pax_fiddle.Config(linears.FeedForward)
    item_fft.copy_fields_from(lm_p.softmax_tpl.feed_forward_tpl)
    item_fft.input_dims = self.ITEM_EMBEDDING_SIZE
    item_fft.output_dims = self.ITEM_VOCAB_SIZE
    # item_fft.linear.theta.w loaded from pretrained embedding checkpoint.
    embedding_softmax_p.item_feed_forward_tpl = item_fft
    
    cluster_fft = pax_fiddle.Config(linears.FeedForward)
    cluster_fft.copy_fields_from(lm_p.softmax_tpl.feed_forward_tpl)
    cluster_fft.input_dims = self.ITEM_EMBEDDING_SIZE
    cluster_fft.output_dims = self.NUM_ITEM_CLUSTERS
    # cluster_fft.linear.theta.w loaded from pretrained embedding checkpoint.
    embedding_softmax_p.cluster_feed_forward_tpl = cluster_fft

    embedding_softmax_p.item_output_dnn_tpl = self._item_output_dnn()
    embedding_softmax_p.item_output_dnn_tpl.mesh_axis_names = (
        self.MESH_AXIS_NAMES
    )
    embedding_softmax_p.item_output_dnn_tpl.ici_mesh_shape = self.ICI_MESH_SHAPE
    
    embedding_softmax_p.item_input_dnn_tpl = self._item_input_dnn()
    embedding_softmax_p.item_input_dnn_tpl.mesh_axis_names = self.MESH_AXIS_NAMES
    embedding_softmax_p.item_input_dnn_tpl.ici_mesh_shape = self.ICI_MESH_SHAPE
    embedding_softmax_p.use_item_input_dnn_everywhere = True
    embedding_softmax_p.correction = self.CORRECTION

    lm_p.softmax_tpl = embedding_softmax_p

    model_p = pax_fiddle.Config(
        eap_lib.EmbeddingInterleavedLanguageModel, name='eilm'
    )
    model_p.model_type = transformer_models.LanguageModelType.CAUSAL
    model_p.lm_tpl = lm_p
    model_p.lm_tpl.model_type = transformer_models.LanguageModelType.CAUSAL

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE

    model_p.mesh_axis_names = self.MESH_AXIS_NAMES
    model_p.ici_mesh_shape = self.ICI_MESH_SHAPE
    model_p.dcn_mesh_shape = None

    # decoding configs
    model_p.decoder_tpl = self._decoding_configs()
    model_p.decoder_tpl.fprop_for_prefix = True
    logging.info('model intitializing complete')
    return model_p

  def _item_input_dnn(self):
    return pax_fiddle.Config(
        linears.MLPBlock,
        fprop_dtype=self.FPROP_DTYPE,
        dtype=self.MODEL_DTYPE,
        hidden_dims=self.ITEM_DNN_MLP_HIDDEN_DIMS,
        num_layers=self.ITEM_DNN_MLP_NUM_LAYERS,
        activate_final=True,
        ff_tpl=pax_fiddle.Config(
            linears.FeedForward,
            input_dims=self.ITEM_EMBEDDING_SIZE,
            output_dims=self.LM_TYPE().task().model.lm_tpl.model_dims,
            has_bias=True,
            activation_tpl=pax_fiddle.Config(activations.GELU),
            weight_init=base_layer.WeightInit.Xavier(1.000001),
            checkpoint_str='item_input_dnn',
        ),
    )

  def _item_output_dnn(self):
    return pax_fiddle.Config(
        linears.MLPBlock,
        fprop_dtype=self.FPROP_DTYPE,
        dtype=self.MODEL_DTYPE,
        hidden_dims=self.ITEM_DNN_MLP_HIDDEN_DIMS,
        num_layers=self.ITEM_DNN_MLP_NUM_LAYERS,
        activate_final=True,
        ff_tpl=pax_fiddle.Config(
            linears.FeedForward,
            input_dims=self.LM_TYPE().task().model.lm_tpl.model_dims,
            output_dims=self.ITEM_EMBEDDING_SIZE,
            has_bias=True,
            activation_tpl=pax_fiddle.Config(activations.GELU),
            weight_init=base_layer.WeightInit.Xavier(1.000001),
            checkpoint_str='item_output_dnn',
        ),
    )

  def _decoding_configs(self, eos_id=1):
    if self.DECODE_ALGORITHM == 'greedy':
      return decoder_hparams.GreedyDecoderHParams(
          seqlen=self.DECODE_SEQ_LEN,
          max_decode_steps=self.MAX_DECODE_STEPS,
          min_prefix_len=0,
          eos_id=eos_id,
      )
    elif self.DECODE_ALGORITHM == 'beam_search':
      return decoder_hparams.BeamSearchHParams(
          beam_size=4,
          length_norm_alpha=1.0,
          eos_id=eos_id,
          seqlen=self.DECODE_SEQ_LEN,
          max_decode_steps=self.MAX_DECODE_STEPS,
      )
    elif self.DECODE_ALGORITHM == 'sample':
      return decoder_hparams.SampleDecoderHParams(
          seqlen=self.DECODE_SEQ_LEN,
          max_decode_steps=self.MAX_DECODE_STEPS,
          min_prefix_len=0,
          eos_id=eos_id,
          k=40,
          num_samples=1,
          temperature=0.8,
      )
    else:
      raise NotImplementedError(self.DECODE_ALGORITHM)


@experiment_registry.register()
class ItemDecoderPALM128MGames(ItemDecoderPALMBase):
  """Item decoder finetuning based on PALM-1B checkpoint."""

  LM_TYPE = PALM128M
  LM_CKPT_PATH: str = (
      '.../ulm/ulm_128M/'
  )
  LM_CKPT_STEP: int = 104_000  # 626_000

  WARMUP_STEPS: int = 1000
  MAX_STEPS: int = 100_000
  LEARNING_RATE_DECAY_END: int = 10_000
  LEARNING_RATE: float = 1e-2
  WEIGHT_DECAY: float = 1e-5
  INPUT_DROPOUT_PROB: float = 0.0
  DROPOUT_PROB: float = 0.0
  PERCORE_BATCH_SIZE: int = 32
  PERCORE_EVAL_BATCH_SIZE: int = 32
  LABEL_SMOOTHING: float = 0.0
  SECOND_LABEL_SMOOTHING: float = 0.0
  TRAINABLE_CLUSTER_EMBEDDINGS: bool = True
  TRAINABLE_ITEM_EMBEDDINGS: bool = True
  FULL_SOFTMAX: bool = True
  CLIP_GRADIENT_NORM_TO_VALUE: float = 1.0
  LAYERWISE_ADAPTATION: bool = False
  SOFT_CAP_LOGITS: float = 30.0

  SAVE_INTERVAL_STEPS: int = 1000
  EVAL_INTERVAL_STEPS: int = 1000

  ICI_MESH_SHAPE: list[int] = [1, 16, 1]

  # Item projection params
  ITEM_DNN_MLP_HIDDEN_DIMS: int = 512
  ITEM_DNN_MLP_NUM_LAYERS: int = 1

  # clustering params
  ITEM_VOCAB_SIZE: int = 16859
  NUM_ITEM_CLUSTERS: int = 286
  CLUSTERING_PKL: str = ''

  # Input params
  MAX_NUM_ITEMS: int = 21  # 22 + 1
  MAX_INPUT_LEN: int = 41  # 59 + 1
  ITEM_EMBEDDING_SIZE: int = 512  # WALS item embedding size.
  ITEM_EMBEDDING_CKPT: str = ''
  CLUSTER_EMBEDDING_CKPT: str = ''
  TASK_NAME: str = 'id_games_ials'
  EVAL_TASK_NAMES: list[str] = ['id_games_ials']
  EVAL_SPLIT_NAMES: list[str] = ['test', 'eval_test']
  DECODE_TASK_NAME: str = 'id_games_ials'

@experiment_registry.register()
class ItemDecoderPALM128MGamesSpectral(ItemDecoderPALM128MGames):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 286
  CLUSTERING_PKL: str = ''
  MAX_NUM_ITEMS: int = 22  # 22 + 1
  MAX_INPUT_LEN: int = 95  # 59 + 1

@experiment_registry.register()
class ItemDecoderPALM8BGamesTasks(ItemDecoderPALM128MGamesSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  LM_TYPE = PALM8B
  LM_CKPT_PATH: str = '.../ulm/ulm_8B/'
  LM_CKPT_STEP: int = 217_800
  PERCORE_BATCH_SIZE: int = 4
  PERCORE_EVAL_BATCH_SIZE: int = 4
  TASK_NAME: str = 'id_games_ials_tasks_pack'
  EVAL_TASK_NAMES: list[str] = ['id_games_ials_seqrec_test']
  DECODE_TASK_NAME: str = 'id_games_ials_seqrec_test'
  MAX_NUM_ITEMS: int = 22  # 22 + 1
  MAX_INPUT_LEN: int = 900  # 200
  ENABLE_SEQUENCE_PACKING: bool = True

@experiment_registry.register()
class ItemDecoderPALM128MOfficeSpectral(ItemDecoderPALM128MGames):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  TASK_NAME: str = 'id__office_ials_tasks_mix'
  EVAL_TASK_NAMES: list[str] = ['id__office_ials_v2']
  DECODE_TASK_NAME: str = 'id__office_ials_v2'
  ITEM_VOCAB_SIZE: int = 27931
  NUM_ITEM_CLUSTERS: int = 2793
  CLUSTERING_PKL: str = (
      '.../Office_frequency_2793.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_frequency_2793/'
  )
  ITEM_EMBEDDING_CKPT: str = (
      '.../Office_random_512/'
  )
  ITEM_EMBEDDING_SIZE: int = 512  # WALS item embedding size.
  MAX_NUM_ITEMS: int = 52  # 22 + 1
  MAX_INPUT_LEN: int = 125  # 59 + 1
  ENABLE_SEQUENCE_PACKING: bool = False

@experiment_registry.register()
class ItemDecoderPALM8BOffice(ItemDecoderPALM128MOfficeSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  LM_TYPE = PALM8B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_8B/'
  LM_CKPT_STEP: int = 217_800
  PERCORE_BATCH_SIZE: int = 8
  PERCORE_EVAL_BATCH_SIZE: int = 8
  MAX_INPUT_LEN: int = 200  # 125

@experiment_registry.register()
class ItemDecoderPALM8BOfficePack(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_office_ials_tasks_pack_recformer'
  EVAL_TASK_NAMES: list[str] = ['id_office_ials_seqrec_test_recformer']
  DECODE_TASK_NAME: str = 'id_office_ials_seqrec_test_recformer'

@experiment_registry.register()
class ItemDecoderPALM8BOfficePackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_office_ials_tasks_pack_calrec2'
  EVAL_TASK_NAMES: list[str] = ['id_office_ials_seqrec_test_calrec2']
  DECODE_TASK_NAME: str = 'id_office_ials_seqrec_test_calrec2'
  MAX_INPUT_LEN: int = 460  # 360
  PERCORE_BATCH_SIZE: int = 32
  PERCORE_EVAL_BATCH_SIZE: int = 32
  ITEM_EMBEDDING_CKPT: str = '.../Office_random_512_2/'
  ITEM_VOCAB_SIZE: int = 27886
  ITEM_EMBEDDING_SIZE: int = 512  # WALS item embedding size.
  ITEM_DNN_MLP_HIDDEN_DIMS: int = 512

@experiment_registry.register()
class ItemDecoderPALM1BOfficePackCalrec(ItemDecoderPALM8BOfficePackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000

@experiment_registry.register()
class ItemDecoderPALM8BOfficePackCalrecSpectral(ItemDecoderPALM8BOfficePackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 455
  CLUSTERING_PKL: str = (
      '.../Office_spectral.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_spectral/'
  )
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM8BOfficePackCalrecFrequency(ItemDecoderPALM8BOfficePackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Office_frequency.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_frequency/'
  )
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BOfficePackCalrecSpectral(ItemDecoderPALM8BOfficePackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BOfficePackCalrecFrequency(ItemDecoderPALM1BOfficePackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Office_frequency.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_frequency/'
  )

@experiment_registry.register()
class ItemDecoderPALM1BOfficePackCalrecRandom(ItemDecoderPALM1BOfficePackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Office_random.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_random/'
  )

@experiment_registry.register()
class ItemDecoderPALM8BGamesPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_games_ials_tasks_pack_calrec'
  EVAL_TASK_NAMES: list[str] = ['id_games_ials_seqrec_test_calrec']
  DECODE_TASK_NAME: str = 'id_games_ials_seqrec_test_calrec'
  MAX_INPUT_LEN: int = 970  # 847
  PERCORE_BATCH_SIZE: int = 16
  PERCORE_EVAL_BATCH_SIZE: int = 16
  ITEM_EMBEDDING_CKPT: str = '.../Games_random_128/'
  ITEM_EMBEDDING_SIZE: int = 128  # WALS item embedding size.
  ITEM_DNN_MLP_HIDDEN_DIMS: int = 128
  ITEM_VOCAB_SIZE: int = 17383

@experiment_registry.register()
class ItemDecoderPALM1BGamesPackCalrec(ItemDecoderPALM8BGamesPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000

@experiment_registry.register()
class ItemDecoderPALM8BGamesPackCalrecSpectral(ItemDecoderPALM8BGamesPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 213
  CLUSTERING_PKL: str = (
      '.../Games_spectral.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Games_spectral/'
  )
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM1BGamesPackCalrecSpectral(ItemDecoderPALM8BGamesPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BGamesPackCalrecFrequency(ItemDecoderPALM1BGamesPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Games_frequency.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Games_frequency/'
  )

@experiment_registry.register()
class ItemDecoderPALM1BGamesPackCalrecRandom(ItemDecoderPALM1BGamesPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Games_random.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Games_random/'
  )

@experiment_registry.register()
class ItemDecoderPALM8BInstrumentsPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_instruments_ials_tasks_pack_calrec2'
  EVAL_TASK_NAMES: list[str] = ['id_instruments_ials_seqrec_test_calrec2']
  DECODE_TASK_NAME: str = 'id_instruments_ials_seqrec_test_calrec2'
  MAX_INPUT_LEN: int = 450  # 312
  PERCORE_BATCH_SIZE: int = 16
  PERCORE_EVAL_BATCH_SIZE: int = 16
  ITEM_EMBEDDING_CKPT: str = '.../Instruments_random_256/'
  ITEM_EMBEDDING_SIZE: int = 256  # WALS item embedding size.
  ITEM_DNN_MLP_HIDDEN_DIMS: int = 256
  
  ITEM_VOCAB_SIZE: int = 10599

@experiment_registry.register()
class ItemDecoderPALM1BInstrumentsPackCalrec(ItemDecoderPALM8BInstrumentsPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000

@experiment_registry.register()
class ItemDecoderPALM128MInstrumentsPackCalrec(ItemDecoderPALM8BInstrumentsPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM128M
  
  LM_CKPT_PATH: str = (
      '.../ulm/ulm_128M/'
  )
  LM_CKPT_STEP: int = 104_000  # 626_000

@experiment_registry.register()
class ItemDecoderPALM8BInstrumentsPackCalrecSpectral(ItemDecoderPALM8BInstrumentsPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 205
  CLUSTERING_PKL: str = (
      '.../Instruments_spectral.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Instruments_spectral/'
  )
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM1BInstrumentsPackCalrecSpectral(ItemDecoderPALM8BInstrumentsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BInstrumentsPackCalrecRandom(ItemDecoderPALM1BInstrumentsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Instruments_random.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Instruments_random/'
  )

@experiment_registry.register()
class ItemDecoderPALM1BInstrumentsPackCalrecFrequency(ItemDecoderPALM1BInstrumentsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Instruments_frequency.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Instruments_frequency/'
  )

@experiment_registry.register()
class ItemDecoderPALM8BArtsPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_arts_ials_tasks_pack_calrec'
  EVAL_TASK_NAMES: list[str] = ['id_arts_ials_seqrec_test_calrec']
  DECODE_TASK_NAME: str = 'id_arts_ials_seqrec_test_calrec'
  MAX_INPUT_LEN: int = 350  # 265
  PERCORE_BATCH_SIZE: int = 16
  PERCORE_EVAL_BATCH_SIZE: int = 16
  ITEM_EMBEDDING_CKPT: str = '.../Arts_random_512/'
  ITEM_EMBEDDING_SIZE: int = 512  # WALS item embedding size.
  ITEM_DNN_MLP_HIDDEN_DIMS: int = 512
  ITEM_VOCAB_SIZE: int = 22828

@experiment_registry.register()
class ItemDecoderPALM8BArtsPackCalrecSpectral(ItemDecoderPALM8BArtsPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 344
  CLUSTERING_PKL: str = (
      '.../Arts_spectral.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Arts_spectral/'
  )
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM1BArtsPackCalrec(ItemDecoderPALM8BArtsPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000

@experiment_registry.register()
class ItemDecoderPALM1BArtsPackCalrecSpectral(ItemDecoderPALM8BArtsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BArtsPackCalrecRandom(ItemDecoderPALM1BArtsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Arts_random.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Arts_random/'
  )

@experiment_registry.register()
class ItemDecoderPALM1BArtsPackCalrecFrequency(ItemDecoderPALM1BArtsPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Arts_frequency.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Arts_frequency/'
  )

@experiment_registry.register()
class ItemDecoderPALM8BScientificPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_scientific_ials_tasks_pack_calrec'
  EVAL_TASK_NAMES: list[str] = ['id_scientific_ials_seqrec_test_calrec']
  DECODE_TASK_NAME: str = 'id_scientific_ials_seqrec_test_calrec'
  MAX_INPUT_LEN: int = 200  # 149
  PERCORE_BATCH_SIZE: int = 8
  PERCORE_EVAL_BATCH_SIZE: int = 8
  ITEM_EMBEDDING_CKPT: str = '.../Scientific_random_512/'
  ITEM_VOCAB_SIZE: int = 5282

@experiment_registry.register()
class ItemDecoderPALM8BScientificPackCalrecSpectral(ItemDecoderPALM8BScientificPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 155
  CLUSTERING_PKL: str = (
      '.../Scientific_spectral.pickle'
  )
  #.../Office_random_2793.pickle
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Scientific_spectral/'
  )
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM8BPetPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_pet_ials_tasks_pack_calrec'
  EVAL_TASK_NAMES: list[str] = ['id_pet_ials_seqrec_test_calrec']
  DECODE_TASK_NAME: str = 'id_pet_ials_seqrec_test_calrec'
  MAX_INPUT_LEN: int = 600  # 542
  PERCORE_BATCH_SIZE: int = 8
  PERCORE_EVAL_BATCH_SIZE: int = 8
  ITEM_EMBEDDING_CKPT: str = '.../Pet_random_512/'
  ITEM_VOCAB_SIZE: int = 42495

@experiment_registry.register()
class ItemDecoderPALM1BPetPackCalrec(ItemDecoderPALM8BPetPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000

@experiment_registry.register()
class ItemDecoderPALM8BPetPackCalrecSpectral(ItemDecoderPALM8BPetPackCalrec):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 508
  CLUSTERING_PKL: str = (
      '.../Pet_spectral.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Pet_spectral/'
  )
  FULL_SOFTMAX: bool = False
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BPetPackCalrecSpectral(ItemDecoderPALM8BPetPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  LM_TYPE = PALM1B
  
  LM_CKPT_PATH: str = '.../ulm/ulm_1B/'
  LM_CKPT_STEP: int = 626_000
  CORRECTION: bool = True

@experiment_registry.register()
class ItemDecoderPALM1BPetPackCalrecRandom(ItemDecoderPALM1BPetPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Pet_random.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Pet_random/'
  )

@experiment_registry.register()
class ItemDecoderPALM1BPetPackCalrecFrequency(ItemDecoderPALM1BPetPackCalrecSpectral):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  CLUSTERING_PKL: str = (
      '.../Pet_frequency.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Pet_frequency/'
  )

@experiment_registry.register()
class ItemDecoderPALM8BFusionPackCalrec(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  ENABLE_SEQUENCE_PACKING: bool = True
  TASK_NAME: str = 'id_fusion_office_ials_tasks_pack_calrec'
  EVAL_TASK_NAMES: list[str] = ['id_fusion_ials_seqrec_test_calrec']
  DECODE_TASK_NAME: str = 'id_fusion_ials_seqrec_test_calrec'
  MAX_INPUT_LEN: int = 400  # 312
  PERCORE_BATCH_SIZE: int = 8
  PERCORE_EVAL_BATCH_SIZE: int = 8
  ITEM_EMBEDDING_CKPT: str = '.../Office_random_512/'
  ITEM_VOCAB_SIZE: int = 27886

@experiment_registry.register()
class ItemDecoderPALM8BOfficeFrequency(ItemDecoderPALM8BOffice):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""
  FULL_SOFTMAX: bool = False

@experiment_registry.register()
class ItemDecoderPALM8BOfficeSpectral(ItemDecoderPALM8BOfficeFrequency):
  """Item decoder finetuning based on PALM-1B checkpoint for Games with Random clustering."""

  NUM_ITEM_CLUSTERS: int = 570
  CLUSTERING_PKL: str = (
      '.../Office_spectral_150.pickle'
  )
  CLUSTER_EMBEDDING_CKPT: str = (
      '.../Office_spectral_570/'
  )
  PERCORE_BATCH_SIZE: int = 4
  PERCORE_EVAL_BATCH_SIZE: int = 4
