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

"""QK-Clip under layer scan: every layer of a scanned block is clipped by its own max logit.

The scan stacks a block's max_logits on axis 0 ([layers, batch, heads]) and its kernels carry the layer axis at
param_scan_axis ([rank, layers, heads, dim]). The scale must be min(1, tau / max logit) per layer and head, as
QK-Clip is defined, not one scale per head taken over all layers of the scan.
"""

import os
import unittest
from collections import namedtuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.utils.qk_clip_utils import apply_qk_clip, apply_qk_clip_nnx

Config = namedtuple("Config", ["qk_clip_threshold", "qk_nope_head_dim", "attention_type", "param_scan_axis"])


class _Box:

  def __init__(self, **kw):
    self.__dict__.update(kw)


class _Attn(nnx.Module):

  def __init__(self, wq_b, wkv_b):
    self.wq_b = _Kernel(wq_b)
    self.wkv_b = _Kernel(wkv_b)


class _Kernel(nnx.Module):

  def __init__(self, w):
    self.kernel = nnx.Param(w)


class _Layer(nnx.Module):

  def __init__(self, attn):
    self.self_attention = attn


class _Decoder(nnx.Module):

  def __init__(self, layer):
    self.moe_layers = layer


class _Model(nnx.Module):

  def __init__(self, decoder):
    self.decoder = decoder


def _expected(w, max_logits, tau, nope, which):
  """Reference: per layer l and head h, scale = min(1, tau / max_b logits[l, b, h])."""
  s = np.minimum(1.0, tau / (np.asarray(max_logits).max(axis=1) + 1e-6))  # [layers, heads]
  out = np.array(w, dtype=np.float64)
  for l in range(s.shape[0]):
    for h in range(s.shape[1]):
      out[:, l, h, :nope] *= np.sqrt(s[l, h])
      if which == "wq_b":
        out[:, l, h, nope:] *= s[l, h]
  return out


class QKClipScanTest(unittest.TestCase):

  def setUp(self):
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    self.rank, self.layers, self.heads, self.nope, self.rope = 3, 3, 2, 4, 2
    self.wq = jax.random.normal(k1, (self.rank, self.layers, self.heads, self.nope + self.rope))
    self.wkv = jax.random.normal(k2, (self.rank, self.layers, self.heads, self.nope + 5))
    # Different max logits per layer: layer 0 head 0 is the largest overall.
    self.ml = jnp.array([[[400.0, 50.0]], [[200.0, 300.0]], [[90.0, 120.0]]])  # [layers, batch=1, heads]
    self.cfg = Config(qk_clip_threshold=100.0, qk_nope_head_dim=self.nope, attention_type="mla", param_scan_axis=1)

  def test_nnx_per_layer_scale(self):
    model = _Model(_Decoder(_Layer(_Attn(self.wq, self.wkv))))
    inter = {"decoder": {"moe_layers": {"self_attention": {"attention_op": {"max_logits": self.ml}}}}}
    apply_qk_clip_nnx(SimpleNamespace(model=model), inter, self.cfg)
    attn = model.decoder.moe_layers.self_attention
    np.testing.assert_allclose(attn.wq_b.kernel.value, _expected(self.wq, self.ml, 100.0, self.nope, "wq_b"), rtol=1e-6)
    np.testing.assert_allclose(
        attn.wkv_b.kernel.value, _expected(self.wkv, self.ml, 100.0, self.nope, "wkv_b"), rtol=1e-6
    )
    # Layer 2 head 0 (max logit 90 < tau) is untouched, though layer 0 head 0 is clipped.
    np.testing.assert_array_equal(attn.wq_b.kernel.value[:, 2, 0], self.wq[:, 2, 0])

  def test_linen_per_layer_scale(self):
    params = {"decoder": {"moe_layers": {"self_attention": {"wq_b": {"kernel": self.wq}, "wkv_b": {"kernel": self.wkv}}}}}
    state = _Box(params=params, replace=lambda params: _Box(params=params))
    inter = {"intermediates": {"decoder": {"moe_layers": {"self_attention": {"max_logits": (self.ml,)}}}}}
    new = apply_qk_clip(state, inter, self.cfg).params["decoder"]["moe_layers"]["self_attention"]
    np.testing.assert_allclose(new["wq_b"]["kernel"], _expected(self.wq, self.ml, 100.0, self.nope, "wq_b"), rtol=1e-6)

  def test_mismatched_layer_axis_raises(self):
    model = _Model(_Decoder(_Layer(_Attn(self.wq, self.wkv))))
    inter = {"decoder": {"moe_layers": {"self_attention": {"attention_op": {"max_logits": self.ml[:2]}}}}}
    with self.assertRaisesRegex(ValueError, "does not match max_logits"):
      apply_qk_clip_nnx(SimpleNamespace(model=model), inter, self.cfg)


class QKClipScanHy4TinyTest(unittest.TestCase):
  """hy4-tiny with 1 dense + 10 MoE layers: IndexCache blocks of 4, scan length 2, plus a remainder block of 2."""

  def test_each_scanned_layer_uses_its_own_scale(self):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from flax.linen import partitioning as nn_partitioning  # pylint: disable=import-outside-toplevel
    from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
    from maxtext.layers.attention_mla import indexer_losses  # pylint: disable=import-outside-toplevel
    from maxtext.utils import maxtext_utils, model_creation_utils  # pylint: disable=import-outside-toplevel
    from maxtext.utils.globals import MAXTEXT_PKG_DIR  # pylint: disable=import-outside-toplevel

    tau = 1e-3  # every head of every layer clips
    cfg = pyconfig.initialize(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "model_name=hy4-tiny",
            "override_model_config=true",
            "base_num_decoder_layers=11",
            "scan_layers=True",
            "attention=dot_product",
            "use_qk_clip=True",
            f"qk_clip_threshold={tau}",
            "dtype=float32",
            "weight_dtype=float32",
            "matmul_precision=highest",
            "per_device_batch_size=1",
            "max_target_length=64",
            "mtp_num_layers=0",
            "enable_dropout=false",
            "skip_jax_distributed_system=True",
            "run_name=qk_clip_scan_test",
        ]
    )
    mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = model_creation_utils.get_nnx_create_model_fn(cfg, mesh)()
      b, t = cfg.global_batch_size_to_train_on, cfg.max_target_length
      tok = jnp.asarray(np.random.default_rng(0).integers(0, cfg.vocab_size, (b, t)), jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t))
      seg = jnp.ones((b, t), jnp.int32)
      model(
          decoder_input_tokens=tok,
          decoder_positions=pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          decoder_target_tokens=tok,
          decoder_target_mask=seg,
      )
      nnx.pop(model, indexer_losses)
      inter = nnx.pop(model, nnx.Intermediate).to_pure_dict()
      blocks = model.decoder.moe_layers
      before = {i: np.asarray(getattr(blocks, f"layers_{i}").self_attention.wq_b.kernel.value) for i in range(4)}
      apply_qk_clip_nnx(SimpleNamespace(model=model), inter, cfg)
      after = {i: np.asarray(getattr(blocks, f"layers_{i}").self_attention.wq_b.kernel.value) for i in range(4)}

    nope = cfg.qk_nope_head_dim
    for i in range(4):
      ml = np.asarray(inter["decoder"]["moe_layers"][f"layers_{i}"]["self_attention"]["attention_op"]["max_logits"])
      self.assertEqual(ml.shape[0], 2)  # scan length 2
      want = np.sqrt(np.minimum(1.0, tau / (ml.max(axis=1) + 1e-6)))  # [layers, heads]
      got = np.median(after[i][..., :nope] / before[i][..., :nope], axis=(0, 3))  # [layers, heads]
      np.testing.assert_allclose(got, want, rtol=1e-5)
      self.assertFalse(np.allclose(want[0], want[1]))  # the two scanned layers really differ


if __name__ == "__main__":
  unittest.main()
