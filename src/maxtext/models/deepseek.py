# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any, Optional

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Config, AttentionType
from maxtext.common.common_types import HyperConnectionType, MODEL_MODE_PREFILL, DecoderBlockType
from maxtext.layers import attention_mla
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.linears import Dropout
from maxtext.layers.engram import Engram
from maxtext.layers.engram import NgramHashMapping
from maxtext.layers.normalizations import RMSNorm
from maxtext.models import deepseek_batchsplit
from maxtext.models import deepseek_batchsplit_fp8
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils.sharding import maybe_shard_with_logical
from maxtext.utils.sharding import get_logical_axis_rules

import transformers

# -----------------------------------------
# The Decoder Layer for DeepSeek v3
# -----------------------------------------


def _indexer_kwargs(shared_indexer_mask, return_indexer_mask, forced_indexer_topk=None):
  """DSA IndexCache kwargs for the attention call, only those that are set (other attention types lack them).

  forced_indexer_topk (debug_forced_indexer_topk): [batch, seq, indexer_topk] key positions; only read by a layer
  that runs its own indexer.
  """
  kwargs = {}
  if forced_indexer_topk is not None:
    kwargs["forced_indexer_topk"] = forced_indexer_topk
  if shared_indexer_mask is not None:
    kwargs["shared_indexer_mask"] = shared_indexer_mask
  if return_indexer_mask:
    kwargs["return_indexer_mask"] = True
  return kwargs


class DeepSeekGenericLayer(nnx.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      use_hyper_connections: Optional[bool] = None,
      has_indexer: Optional[bool] = None,
  ) -> None:
    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    # None: follow mhc_expansion_rate. False: a plain residual layer on a 3D state even in an mHC model
    # (e.g. the generic MTP layer, which runs on the collapsed hidden state).
    if use_hyper_connections and config.mhc_expansion_rate <= 1:
      raise ValueError("use_hyper_connections=True requires mhc_expansion_rate > 1.")
    self.is_mhc_enabled = config.mhc_expansion_rate > 1 if use_hyper_connections is None else use_hyper_connections
    self.layer_idx = layer_idx
    self.is_engram_enabled = config.engram_layers and layer_idx in config.engram_layers

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names, rules=get_logical_axis_rules())
    self.mlp_intermediate_sharding = create_sharding(
        self.mesh, self.mlp_logical_axis_names, rules=get_logical_axis_rules()
    )

    self.pre_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        single_rounding=getattr(self.config, "rmsnorm_single_rounding", False),
        scale_grad_fp32=getattr(self.config, "rmsnorm_scale_grad_fp32", False),
        scale_grad_gpu_order=getattr(self.config, "rmsnorm_scale_grad_gpu_order", False),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    self.post_self_attention_layer_norm = RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        single_rounding=getattr(self.config, "rmsnorm_single_rounding", False),
        scale_grad_fp32=getattr(self.config, "rmsnorm_scale_grad_fp32", False),
        scale_grad_gpu_order=getattr(self.config, "rmsnorm_scale_grad_gpu_order", False),
        epsilon=self.config.normalization_layer_epsilon,
        rngs=rngs,
    )

    if self.is_engram_enabled:
      self.engram_layer_norm = RMSNorm(
          num_features=self.dummy_inputs_shape[-1],
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          kernel_axes=("norm",),
          epsilon=self.config.normalization_layer_epsilon,
          rngs=rngs,
      )
      tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, token=config.hf_access_token)
      # TODO(ranran): Refactor NgramHashMapping to initialize once globally or at the model level.
      # Moving this to decoders.py currently causes JAX initialization errors.
      self.ngram_hash_mapping = NgramHashMapping(
          engram_vocab_bases=config.engram_vocab_bases,
          max_ngram_size=config.engram_max_ngram_size,
          engram_num_heads=config.engram_num_heads,
          layer_ids=config.engram_layers,
          tokenizer=tokenizer,
          pad_id=tokenizer.pad_token_id,
          seed=config.engram_seed,
      )
      self.engram = Engram(
          config=config,
          mesh=mesh,
          vocab_sizes=self.ngram_hash_mapping.get_vocab_sizes(layer_idx),
          engram_num_heads=config.engram_num_heads,
          engram_head_dim=config.engram_head_dim,
          engram_max_ngram_size=config.engram_max_ngram_size,
          engram_kernel_size=config.engram_kernel_size,
          mhc_expansion_rate=config.mhc_expansion_rate,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.engram_layer_norm = None
      self.engram = None

    # DeepSeek V4 natively overrides this block with CompressedAttention.
    if self.config.decoder_block != DecoderBlockType.DEEPSEEK4:
      self.self_attention = attention_mla.MLA(
          config=self.config,
          num_query_heads=self.config.num_query_heads,
          num_kv_heads=self.config.num_kv_heads,
          head_dim=self.config.head_dim,
          max_target_length=self.config.max_target_length,
          max_prefill_predict_length=self.config.max_prefill_predict_length,
          attention_kernel=self.config.attention,
          attention_type=AttentionType(self.config.attention_type),
          inputs_q_shape=self.dummy_inputs_shape,
          inputs_kv_shape=self.dummy_inputs_shape,
          mesh=mesh,
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          dropout_rate=self.config.dropout_rate,
          name="self_attention",
          quant=quant,
          kv_quant=quantizations.configure_kv_quant(self.config),
          q_lora_rank=self.config.q_lora_rank,
          kv_lora_rank=self.config.kv_lora_rank,
          qk_nope_head_dim=self.config.qk_nope_head_dim,
          qk_rope_head_dim=self.config.qk_rope_head_dim,
          v_head_dim=self.config.v_head_dim,
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          mscale=self.config.mscale,
          rope_factor=self.config.rope_factor,
          model_mode=model_mode,
          rngs=rngs,
          attn_logits_soft_cap=self.config.attn_logits_soft_cap,
          has_indexer=has_indexer,
          # dot_product attention only; MLA took the constructor defaults (False) before, so these configs were ignored
          float32_qk_product=self.config.float32_qk_product,
          float32_logits=self.config.float32_logits,
      )

    self.dropout = Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)
    if self.is_mhc_enabled:
      self.mhc_attention = mhc.make_hyper_connections(self.config, self.config.emb_dim, self.mesh, self.rngs)
      self.mhc_mlp = mhc.make_hyper_connections(self.config, self.config.emb_dim, self.mesh, self.rngs)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    """Executes the MLP operation. To be implemented by subclasses."""
    raise NotImplementedError()

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        rules=get_logical_axis_rules(),
    )

  def dropout_op(self, x, deterministic):
    dropout = self.dropout(x, deterministic=deterministic)
    return self.with_logical_constraint(dropout)

  def pre_attention_norm_op(self, x):
    pre_attention_norm = self.pre_self_attention_layer_norm(x)
    return self.with_logical_constraint(pre_attention_norm)

  def post_attention_norm_op(self, x):
    post_attention_norm = self.post_self_attention_layer_norm(x)
    return self.with_logical_constraint(post_attention_norm)

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      **indexer_kwargs,
  ):
    """Executes the attention layer.

    indexer_kwargs (DSA IndexCache, MLA only): shared_indexer_mask, return_indexer_mask. With
    return_indexer_mask=True this returns (attention_result, indexer_mask).
    """
    attention_outputs = self.self_attention(
        x,
        x,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        out_sharding=self.out_sharding,
        previous_chunk=previous_chunk,
        slot=slot,
        **indexer_kwargs,
    )
    attention_result = self.with_logical_constraint(attention_outputs[0])
    if indexer_kwargs.get("return_indexer_mask"):
      return attention_result, attention_outputs[2]
    return attention_result

  @property
  def logical_axis_names(self):
    """Generate logical names for activations generally."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_embed"]
    return axis_names

  @property
  def mlp_logical_axis_names(self):
    """Generate logical names for activations in MLP."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    axis_names = ["activation_batch", length_name, "activation_mlp"]
    return axis_names

  def post_process(self, layer_output, load_balance_loss, moe_bias_updates, kv_cache=None):
    """postprocessing."""

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.te_moe_block and moe_bias_updates is not None:
      total_recv_tokens, recv_capacity_per_rank = moe_bias_updates
      self.sow(nnx.Intermediate, "te_moe_total_recv_tokens", total_recv_tokens)
      self.sow(nnx.Intermediate, "te_moe_recv_capacity_per_rank", recv_capacity_per_rank)
      self.sow(
          nnx.Intermediate,
          "te_moe_capacity_overflow",
          jnp.any(total_recv_tokens > recv_capacity_per_rank),
      )
    elif self.config.routed_bias and self.config.routed_bias_update_rate > 0.0 and moe_bias_updates is not None:
      self.sow(nnx.Intermediate, "moe_bias_updates", moe_bias_updates)

    if getattr(self.config, "record_internal_nn_metrics", False):
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache

  def self_attention_with_norm_op(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      **indexer_kwargs,
  ):
    """self-attention with normalization.

    indexer_kwargs (DSA IndexCache): shared_indexer_mask, return_indexer_mask; with return_indexer_mask=True the
    attention layer's top-k mask is returned as a third output.
    """
    indexer_mask = None
    if self.is_mhc_enabled:
      intermediate_inputs, attention_metadata = self.mhc_attention(
          self.pre_attention_norm_op,
          self.self_attention,
          x=inputs,
          mhc_type=HyperConnectionType.ATTENTION,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=deterministic,
          model_mode=model_mode,
          out_sharding=self.out_sharding,
          previous_chunk=previous_chunk,
          slot=slot,
          **indexer_kwargs,
      )
      indexer_mask = attention_metadata.get("indexer_mask")
    else:
      lnx = self.pre_attention_norm_op(inputs)
      attention_lnx = self.attention_op(
          lnx,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          **indexer_kwargs,
      )
      if indexer_kwargs.get("return_indexer_mask"):
        attention_lnx, indexer_mask = attention_lnx
      intermediate_inputs = inputs + attention_lnx
    # Normalization. With hyper-connections the MLP branch normalizes its own iHC/mHC-mixed input (mhc_mlp is handed
    # post_attention_norm_op), so this result is unused there; skip it, since flat streams (use_ihc_pallas_kernel,
    # (batch, seq, k * emb)) do not end in emb.
    hidden_states = None if self.is_mhc_enabled else self.post_attention_norm_op(intermediate_inputs)
    if indexer_kwargs.get("return_indexer_mask"):
      return hidden_states, intermediate_inputs, indexer_mask
    return hidden_states, intermediate_inputs

  def engram_op(self, x, decoder_input_tokens):
    normed_x = self.engram_layer_norm(x)  # pyrefly: ignore[not-callable]
    hash_ids = self.ngram_hash_mapping(decoder_input_tokens)[self.layer_idx]
    return self.engram(normed_x, hash_ids)  # pyrefly: ignore[not-callable]


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  """DeepSeek-style dense layer with Multi-Head Latent Attention."""

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      use_hyper_connections: Optional[bool] = None,
      has_indexer: Optional[bool] = None,
  ) -> None:
    super().__init__(
        config,
        model_mode,
        mesh,
        rngs,
        quant,
        layer_idx,
        use_hyper_connections=use_hyper_connections,
        has_indexer=has_indexer,
    )
    self.mlp = linears.MlpBlock(
        in_features=self.dummy_inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=quant,
        model_mode=model_mode,
        mesh=mesh,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic, *args, **kwargs):
    mlp = self.mlp(x, deterministic, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding)
    return self.with_logical_constraint(mlp)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      shared_indexer_mask=None,
      return_indexer_mask: bool = False,
      forced_indexer_topk=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    indexer_kwargs = _indexer_kwargs(shared_indexer_mask, return_indexer_mask, forced_indexer_topk)
    attention_outputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
        **indexer_kwargs,
    )
    hidden_states, intermediate_inputs = attention_outputs[:2]

    if self.is_mhc_enabled:
      layer_output, _ = self.mhc_mlp(
          self.post_attention_norm_op,
          self.mlp,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_DENSE,
          deterministic=deterministic,
      )
    else:
      mlp_lnx = self.mlp_op(hidden_states, deterministic)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    outputs = self.post_process(layer_output, None, None, kv_cache)
    return (*outputs, attention_outputs[2]) if return_indexer_mask else outputs


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekMoELayer(DeepSeekGenericLayer):
  """DeepSeek-style MoE layer with Multi-Head Latent Attention.

  Supports dropless and dropping base on configs. Uses a bias in routing instead
  of load balancing loss.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      use_hyper_connections: Optional[bool] = None,
      has_indexer: Optional[bool] = None,
  ) -> None:
    super().__init__(
        config,
        model_mode,
        mesh,
        rngs,
        quant,
        layer_idx,
        use_hyper_connections=use_hyper_connections,
        has_indexer=has_indexer,
    )
    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      shared_indexer_mask=None,
      return_indexer_mask: bool = False,
      forced_routed_experts=None,
      forced_indexer_topk=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    # Router replay ([batch, seq, top_k]); only reaches here with debug_forced_routing_deepseek (nnx_decoders.py).
    moe_kwargs = {} if forced_routed_experts is None else {"forced_routed_experts": forced_routed_experts}

    # This code should only be traced during initialization when using
    # batch-split schedule. It is never run during model execution, since
    # `Decoder` directly calls `batch_split_schedule` during execution.
    # That is also why we can split/merge activations here as well as
    # in `Decoder`, since they will never be executed together.
    if self.config.use_batch_split_schedule:
      # The older version of batch-split that fully uses qwix quantization.
      if self.config.quantization and self.config.use_qwix_quantization and not self.config.use_manual_quantization:
        activation_pspec = jax.sharding.PartitionSpec(
            ("data", "fsdp", "fsdp_transpose", "expert", "context"),
            None,
            None,
        )
        inputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.split,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=activation_pspec,
            out_specs=[activation_pspec] * self.config.batch_split_factor,
        )(inputs)
        dpos = deepseek_batchsplit_fp8.split(decoder_positions, self.config.batch_split_factor)
        dseg = deepseek_batchsplit_fp8.split(decoder_segment_ids, self.config.batch_split_factor)
        weights = deepseek_batchsplit_fp8.fetch_weights(
            nnx.to_pure_dict(nnx.state(self, (nnx.Param, moe.MoEBiasVar))), self.config.dtype
        )
        outputs = deepseek_batchsplit_fp8.batch_split_schedule(
            inputs,
            weights,
            dpos,
            dseg,
            model_mode=model_mode,
            mesh=self.mesh,
            quant=self.quant,
            cfg=self.config,
        )
        outputs = jax.shard_map(
            functools.partial(
                deepseek_batchsplit_fp8.merge,
                split_factor=self.config.batch_split_factor,
            ),
            mesh=self.mesh,
            in_specs=([activation_pspec] * self.config.batch_split_factor,),
            out_specs=activation_pspec,
        )(outputs)
        return outputs, None

      # bf16 and fp8 code path for pure-JAX batch-split.
      # fp8 code path supports both manual quantization and qwix
      # quantization.
      input_sharding = jax.typeof(inputs).sharding
      activation_pspec = jax.sharding.PartitionSpec(
          ("data", "fsdp", "expert"),
          None,
          None,
      )
      inputs = jax.reshard(inputs, jax.sharding.NamedSharding(self.mesh, activation_pspec))
      yarn_freqs = deepseek_batchsplit.initialize_yarn_freqs(
          decoder_positions,
          embedding_dims=self.config.qk_rope_head_dim,
          rope_theta=self.config.rope_max_timescale,
          max_position_embeddings=self.config.max_position_embeddings,
          original_max_position_embeddings=self.config.original_max_position_embeddings,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_factor=self.config.rope_factor,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      yarn_mask = deepseek_batchsplit.initialize_yarn_mask(self.config.qk_rope_head_dim)
      splash_kernel = deepseek_batchsplit.init_splash_kernel(self.config)
      inputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.split,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=activation_pspec,
          out_specs=[activation_pspec] * self.config.batch_split_factor,
      )(inputs)
      yarn_freqs = deepseek_batchsplit.split(yarn_freqs, self.config.batch_split_factor)

      def extract_fn(x):
        if isinstance(x, nnx.variablelib.Variable):
          return maybe_shard_with_logical(
              x.value,
              x.sharding_names,
              self.mesh,
              shard_mode=self.config.shard_mode,
              rules=get_logical_axis_rules(),
          )
        return x

      weights = deepseek_batchsplit.fetch_weights(
          nnx.to_pure_dict(nnx.state(self, (nnx.Param, moe.MoEBiasVar)), extract_fn), self.config.dtype
      )
      weights = deepseek_batchsplit.gather_weights(weights, self.mesh)
      outputs, _ = deepseek_batchsplit.batch_split_schedule(
          inputs,
          weights,
          yarn_freqs,
          mesh=self.mesh,
          cfg=self.config,
          splash_kernel=splash_kernel,
          activation_pspec=activation_pspec,
          pairwise_swap_and_negate_mask=yarn_mask,
      )
      moe_inputs, routed_expert_out, shared_expert_out, selected_experts = outputs[1]
      outputs[1], _ = deepseek_batchsplit.unroute_ubatch_shard_mapped(
          moe_inputs,
          routed_expert_out,
          shared_expert_out,
          selected_experts,
          expert_axis_name="expert",
          use_gather_mosaic_kernel=False,
          target_length=self.config.max_target_length,
          mesh=self.mesh,
          activation_pspec=activation_pspec,
      )
      outputs = jax.shard_map(
          functools.partial(
              deepseek_batchsplit.merge,
              split_factor=self.config.batch_split_factor,
          ),
          mesh=self.mesh,
          in_specs=([activation_pspec] * self.config.batch_split_factor,),
          out_specs=activation_pspec,
      )(outputs)
      outputs = jax.reshard(outputs, input_sharding)
      return outputs, None

    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    if self.is_engram_enabled:
      engram_output = self.engram_op(x, decoder_input_tokens)
      x = x + engram_output

    indexer_kwargs = _indexer_kwargs(shared_indexer_mask, return_indexer_mask, forced_indexer_topk)
    attention_outputs = self.self_attention_with_norm_op(
        x,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk,
        slot,
        **indexer_kwargs,
    )
    hidden_states, intermediate_inputs = attention_outputs[:2]

    if self.is_mhc_enabled:
      layer_output, metadata = self.mhc_mlp(
          self.post_attention_norm_op,
          self.DeepSeekMoeBlock_0,
          x=intermediate_inputs,
          mhc_type=HyperConnectionType.MLP_MOE,
          **moe_kwargs,
      )
      load_balance_loss = metadata["load_balance_loss"]
      moe_bias_updates = metadata["moe_bias_updates"]
    else:
      mlp_lnx, load_balance_loss, moe_bias_updates = self.mlp_op(hidden_states, deterministic, **moe_kwargs)
      layer_output = mlp_lnx + intermediate_inputs
    layer_output = self.dropout_op(layer_output, deterministic=deterministic)

    outputs = self.post_process(layer_output, load_balance_loss, moe_bias_updates, kv_cache)
    return (*outputs, attention_outputs[2]) if return_indexer_mask else outputs

  def mlp_op(self, x, deterministic, *args, forced_routed_experts=None, **kwargs):
    moe_kwargs = {} if forced_routed_experts is None else {"forced_routed_experts": forced_routed_experts}
    mlp_lnx, load_balance_loss, moe_bias_updates = self.DeepSeekMoeBlock_0(
        x, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding, **moe_kwargs
    )
    return self.with_logical_constraint(mlp_lnx), load_balance_loss, moe_bias_updates


DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekIndexCacheMoEBlock(nnx.Module):
  """`num_of_layers` consecutive DeepSeek MoE layers sharing one DSA top-k selection (IndexCache).

  The first layer runs its own indexer; the following layers have none and attend with the first layer's top-k
  mask. Hy4 groups its MoE layers as 1 indexer layer + (indexer_share_interval - 1) sharing layers. Scanning
  these blocks keeps the mask inside one scan step, so no cross-layer carry is needed.
  """

  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      layer_idx: int = -1,
      num_of_layers: Optional[int] = None,
      remat_policy_fn: Any = None,
      apply_internal_remat: bool = False,
  ) -> None:
    self.config = config
    self.num_of_layers = config.indexer_share_interval if num_of_layers is None else num_of_layers
    self.remat_policy_fn = remat_policy_fn
    self.apply_internal_remat = apply_internal_remat
    for i in range(self.num_of_layers):
      setattr(
          self,
          f"layers_{i}",
          DeepSeekMoELayer(config, model_mode, mesh, rngs, quant, layer_idx, has_indexer=(i == 0)),
      )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      decoder_input_tokens=None,
      forced_routed_experts=None,
      forced_indexer_topk=None,
  ):
    """forced_routed_experts: router replay for the block's layers, [num_of_layers, batch, seq, top_k] (or None).

    forced_indexer_topk: indexer top-k replay, [num_of_layers, batch, seq, indexer_topk]; only the first layer (the
    one with the indexer) reads its slot.
    """
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    if forced_routed_experts is not None and forced_routed_experts.shape[0] != self.num_of_layers:
      raise ValueError(
          f"forced_routed_experts leading axis {forced_routed_experts.shape[0]} != block layers {self.num_of_layers}"
      )
    y, indexer_mask = inputs, None
    for i in range(self.num_of_layers):
      layer = getattr(self, f"layers_{i}")
      fre = None if forced_routed_experts is None else forced_routed_experts[i]

      def run(layer, y, indexer_mask, fre, first=(i == 0)):
        layer_args = (y, decoder_segment_ids, decoder_positions, deterministic, model_mode, previous_chunk, slot)
        kw = {"decoder_input_tokens": decoder_input_tokens}
        if fre is not None:
          kw["forced_routed_experts"] = fre
        if first and forced_indexer_topk is not None:
          kw["forced_indexer_topk"] = forced_indexer_topk[0]
        if first:
          y, _, indexer_mask = layer(*layer_args, return_indexer_mask=True, **kw)
        else:
          y, _ = layer(*layer_args, shared_indexer_mask=indexer_mask, **kw)
        return y, indexer_mask

      if self.apply_internal_remat and self.config.remat_policy != "none":
        y, indexer_mask = self._run_rematted(run, layer, y, indexer_mask, fre)
      else:
        y, indexer_mask = run(layer, y, indexer_mask, fre)
    return y, None

  def _run_rematted(self, run, layer, y, indexer_mask, fre=None):
    """Runs one sub-layer behind its own remat boundary.

    The block body unrolls its sub-layers, so XLA is free to issue every sub-layer's FSDP weight all-gathers
    up front; on hy4-770b that kept three layers of gathered expert weights live and overflowed HBM, while the
    same model scanned one layer at a time fit. Fencing the layer's state on its input pins each sub-layer's
    gathers behind the previous sub-layer's output. The barrier is an identity, so the math is unchanged.
    """
    graphdef, state = nnx.split(layer)

    def pure(state_in, y_in, mask_in, fre_in):
      y_in, state_in = jax.lax.optimization_barrier((y_in, state_in))
      merged = nnx.merge(graphdef, state_in)
      y_out, mask_out = run(merged, y_in, mask_in, fre_in)
      return y_out, mask_out, nnx.state(merged)

    pure = jax.checkpoint(
        pure,
        policy=self.remat_policy_fn,
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config),
    )
    y, indexer_mask, new_state = pure(state, y, indexer_mask, fre)
    nnx.update(layer, new_state)
    return y, indexer_mask
