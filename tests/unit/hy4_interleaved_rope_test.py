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

"""Hy4-preview interleaved q_pe / k_pe layout (MLA and DSA indexer) and serving norm epsilons.

The released checkpoint stores q_pe / k_pe in interleaved pairs (2i, 2i+1). With rope_interleave /
indexer_rope_interleave on, the model must give the same logits as the half-split model run on weights whose
rope columns were de-interleaved; with the switches off on the same weights it must not. hy4-preview uses the
second form: checkpoint_rope_interleave de-interleaves those columns at conversion.
"""

from types import SimpleNamespace
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.hf_model_configs import hy4_tiny_dict
from maxtext.configs import pyconfig
from maxtext.layers.embeddings import YarnRotaryEmbedding
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _config(model_name, **kw):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="hy4_interleaved_rope_test",
      enable_checkpointing=False,
      model_name=model_name,
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=32,
      **kw,
  )


def _deinterleave(x):
  """(2i, 2i+1) pairs -> [even..., odd...] on the last axis."""
  return np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)


# Parameters whose last axis ends with the rope_head_dim q_pe / k_pe columns (MLA wq_b / wkv_a: [nope, rope] and
# [kv_lora, rope]; Hy4 indexer: indexer_rope_position="last").
_MLA_ROPE = ("self_attention/wq_b/kernel", "self_attention/wkv_a/kernel")
_INDEXER_ROPE = ("indexer/wq_b/kernel", "indexer/wk/kernel", "indexer/k_norm/scale", "indexer/k_norm/bias")


class Hy4InterleavedRopeTest(unittest.TestCase):

  def test_preview_config_deinterleaves_at_conversion_and_uses_serving_eps(self):
    cfg = _config("hy4-preview", attention="dot_product")
    # The checkpoint's interleaved q_pe / k_pe columns are de-interleaved at conversion; the model runs concatenated RoPE.
    self.assertFalse(cfg.rope_interleave)
    self.assertFalse(cfg.indexer_rope_interleave)
    self.assertTrue(cfg.checkpoint_rope_interleave)
    self.assertEqual(cfg.mla_lora_norm_epsilon, 1e-5)
    self.assertEqual(cfg.indexer_k_norm_epsilon, 1e-6)
    # hy4-tiny reads the checkpoint the same way (one reading for every Hy4 size).
    tiny = _config("hy4-tiny", attention="dot_product")
    for k in (
        "rope_interleave",
        "indexer_rope_interleave",
        "checkpoint_rope_interleave",
        "mla_lora_norm_epsilon",
        "indexer_k_norm_epsilon",
    ):
      self.assertEqual(getattr(tiny, k), getattr(cfg, k), k)

  def test_yarn_interleave_equals_deinterleave_then_rotate_half(self):
    """YarnRotaryEmbedding(interleave=True)(x) == rotate-half RoPE on de-interleaved x."""
    cfg = _config("hy4-tiny", attention="dot_product")
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    kw = dict(
        embedding_dims=16,
        mesh=mesh,
        max_position_embeddings=64,
        original_max_position_embeddings=64,
        rope_theta=1e7,
        rope_factor=1,
        fprop_dtype=jnp.float32,
        cast_as_fprop_dtype=False,
    )
    x = np.random.default_rng(0).standard_normal((2, 8, 3, 16)).astype(np.float32)
    pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None], (2, 8))
    inter = YarnRotaryEmbedding(interleave=True, **kw)(jnp.asarray(x), position=pos)
    half = YarnRotaryEmbedding(interleave=False, **kw)(jnp.asarray(_deinterleave(x)), position=pos)
    np.testing.assert_array_equal(np.asarray(inter), np.asarray(half))
    # Negative control: reading the same interleaved input as half-split gives something else.
    wrong = YarnRotaryEmbedding(interleave=False, **kw)(jnp.asarray(x), position=pos)
    self.assertGreater(float(np.abs(np.asarray(wrong) - np.asarray(inter)).max()), 1e-2)

  def _build(self, cfg):
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    return model

  def _logits(self, cfg, model, tokens):
    pos = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    seg = jnp.ones_like(tokens)
    with nn_partitioning.axis_rules(cfg.logical_axis_rules):
      out = model(tokens, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode="train")
    out = out[0] if isinstance(out, tuple) else out
    return np.asarray(out, dtype=np.float32)

  def _load(self, model, flat, permute_suffixes, rope_dim, feature_dim):
    """Copies `flat` params into `model`, de-interleaving the last rope_dim features of params matching the suffixes.

    Kernels keep the feature axis last; scanned 1-D params (k_norm scale / bias) put the layer axis last, e.g.
    (indexer_head_dim, 1), so their feature axis is the first one of size feature_dim.
    """
    state = nnx.state(model, nnx.Param)
    new = []
    for path, var in nnx.to_flat_state(state):
      key = "/".join(map(str, path))
      v = np.array(flat[key])
      if key.endswith(permute_suffixes):
        if "k_norm" in key:
          assert list(v.shape).count(feature_dim) == 1, (key, v.shape)  # else the feature axis is ambiguous
          axis = list(v.shape).index(feature_dim)
        else:
          axis = v.ndim - 1
        v = np.moveaxis(v, axis, -1)
        v[..., -rope_dim:] = _deinterleave(v[..., -rope_dim:])
        v = np.moveaxis(v, -1, axis)
      var.set_value(jnp.asarray(v, dtype=var.get_value().dtype))
      new.append((path, var))
    nnx.update(model, nnx.from_flat_state(new))

  def test_model_interleaved_equals_half_split_on_deinterleaved_weights(self):
    kw = dict(weight_dtype="float32", dtype="float32", attention="dot_product", override_model_config=True)
    cfg_i = _config(
        "hy4-tiny", rope_interleave=True, indexer_rope_interleave=True, checkpoint_rope_interleave=False, **kw
    )
    cfg_h = _config("hy4-tiny", **kw)
    rope_dim, feat = cfg_i.qk_rope_head_dim, cfg_i.indexer_head_dim
    # The indexer only changes the logits through its top-k mask, so the indexer negative control needs topk < seq.
    self.assertLess(cfg_i.indexer_topk, cfg_i.max_target_length)
    model_i = self._build(cfg_i)
    rng = np.random.default_rng(1)
    flat = {}
    for path, var in nnx.to_flat_state(nnx.state(model_i, nnx.Param)):
      key = "/".join(map(str, path))
      v = np.asarray(var.get_value())
      if key.endswith(("k_norm/scale", "k_norm/bias")):  # init is ones / zeros, which a permutation cannot detect
        v = (v + 0.5 * rng.standard_normal(v.shape)).astype(v.dtype)
      flat[key] = v
    matched_mla = [k for k in flat if k.endswith(_MLA_ROPE)]
    matched_idx = [k for k in flat if k.endswith(_INDEXER_ROPE)]
    self.assertTrue(matched_mla and matched_idx, sorted(flat))
    self._load(model_i, flat, (), rope_dim, feat)

    tokens = jnp.asarray(rng.integers(0, cfg_i.vocab_size, (1, 32)), dtype=jnp.int32)
    ref = self._logits(cfg_i, model_i, tokens)

    model_h = self._build(cfg_h)
    self._load(model_h, flat, _MLA_ROPE + _INDEXER_ROPE, rope_dim, feat)
    np.testing.assert_allclose(self._logits(cfg_h, model_h, tokens), ref, rtol=1e-5, atol=1e-5)

    # Negative controls: same weights without de-interleaving; MLA de-interleaved but indexer not.
    self._load(model_h, flat, (), rope_dim, feat)
    self.assertGreater(float(np.abs(self._logits(cfg_h, model_h, tokens) - ref).max()), 1e-3)
    self._load(model_h, flat, _MLA_ROPE, rope_dim, feat)
    self.assertGreater(float(np.abs(self._logits(cfg_h, model_h, tokens) - ref).max()), 1e-3)

  def test_conversion_hook_deinterleaves_rope_columns(self):
    """checkpoint_rope_interleave: the HF -> MaxText hooks apply the same permutation the model test above uses."""
    hf_cfg = dict(hy4_tiny_dict)
    rope, heads, nope = hf_cfg["qk_rope_head_dim"], hf_cfg["num_attention_heads"], hf_cfg["qk_nope_head_dim"]
    q_lora, kv_lora, emb = hf_cfg["q_lora_rank"], hf_cfg["kv_lora_rank"], hf_cfg["hidden_size"]
    idx_heads, idx_dim, v_dim = hf_cfg["index_n_heads"], hf_cfg["index_head_dim"], hf_cfg["v_head_dim"]
    layout = dict(use_indexer=True, indexer_share_interval=4, mtp_num_layers=1, first_num_dense_layers=1)
    on = param_mapping.HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(
        hf_cfg, SimpleNamespace(checkpoint_rope_interleave=True, **layout), scan_layers=True
    )
    off = param_mapping.HY4_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, SimpleNamespace(**layout), scan_layers=True)
    rng = np.random.default_rng(3)
    cases = {  # MaxText key suffix: (HF tensor shape, MaxText slice shape)
        "self_attention-wq_b-kernel": ((heads * (nope + rope), q_lora), (q_lora, heads, nope + rope)),
        "self_attention-wkv_a-kernel": ((kv_lora + rope, emb), (emb, kv_lora + rope)),
        "self_attention-indexer-wq_b-kernel": ((idx_heads * idx_dim, q_lora), (q_lora, idx_heads, idx_dim)),
        "self_attention-indexer-wk-kernel": ((idx_dim, emb), (emb, idx_dim)),
        "self_attention-indexer-k_norm-scale": ((idx_dim,), (idx_dim,)),
        "self_attention-indexer-k_norm-bias": ((idx_dim,), (idx_dim,)),
    }
    for suffix, (hf_shape, mt_shape) in cases.items():
      keys = [k for k in on if k.endswith("-" + suffix)]
      stacks = {k.split("-self_attention-")[0] for k in keys}
      # Every stack that carries the tensor: dense, both IndexCache stacks with an indexer or all MoE stacks, MTP.
      self.assertTrue(any("dense_layers" in k for k in stacks) and any("mtp_block" in k for k in stacks), stacks)
      self.assertTrue(any("moe_layers" in k for k in stacks), stacks)
      x = rng.standard_normal(hf_shape).astype(np.float32)
      for key in keys:
        plain = off.get(key)
        base = np.asarray(plain(x, mt_shape) if plain is not None else x)
        expect = base.copy()
        expect[..., -rope:] = _deinterleave(base[..., -rope:])
        got = np.asarray(on[key](x, mt_shape))
        np.testing.assert_array_equal(got, expect, err_msg=key)
        self.assertGreater(float(np.abs(got - base).max()), 0.0, key)  # the permutation is not the identity
        np.testing.assert_array_equal(got[..., :-rope], base[..., :-rope], err_msg=key)  # nope columns untouched
      # The permutation also works on JAX arrays (the dynamic safetensors loader passes them in).
      got_jax = np.asarray(on[keys[0]](jnp.asarray(x), mt_shape))
      np.testing.assert_array_equal(got_jax, np.asarray(on[keys[0]](x, mt_shape)), err_msg=suffix)
    # Parameters outside q_pe / k_pe keep the plain transform.
    key = next(k for k in on if k.endswith("-self_attention-wkv_b-kernel"))
    x = rng.standard_normal((heads * (nope + v_dim), kv_lora)).astype(np.float32)
    shape = (kv_lora, heads, nope + v_dim)
    np.testing.assert_array_equal(np.asarray(on[key](x, shape)), np.asarray(off[key](x, shape)))

  def test_checkpoint_rope_interleave_validation(self):
    """It needs both runtime interleave switches off, and only the Hy4 converter implements it."""
    kw = dict(attention="dot_product", override_model_config=True)
    for bad in (dict(rope_interleave=True), dict(indexer_rope_interleave=True)):
      with self.assertRaises(Exception):
        _config("hy4-preview", **bad, **kw)
    with self.assertRaises(Exception):
      _config("deepseek3-test", checkpoint_rope_interleave=True, rope_interleave=False, **kw)


if __name__ == "__main__":
  unittest.main()
