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

"""Precision switches on the Hy4 path (deepseek decoder + MLA + DSA indexer + MoE + iHC + MTP), with hy4-tiny.

The module-level properties of each switch are tested in its own file (rmsnorm_single_rounding_test,
dense_fp32_accumulation_test, mlp_activation_round_per_op_test, router_gate_bf16_rounding_test,
grad_fp32_switches_test). This file checks that each switch reaches the Hy4 modules it is meant to cover, and
nothing else (iHC and the indexer stay as they are), and that a forward/backward through the whole tiny model runs
with the switches on and changes the numerics.
"""

import sys
import types
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.layers import linears, moe
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path

BF16, F32 = jnp.bfloat16, jnp.float32

SWITCHES = {
    "rmsnorm_single_rounding": True,
    "rmsnorm_scale_grad_fp32": True,
    "dense_fp32_accumulation": True,
    "mlp_activation_round_per_op": True,
    "routed_gate_bf16_rounding": True,
    "logits_dense_wgrad_fp32": True,
    "embedding_grad_fp32": True,
}


def _config(**kw):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="hy4_precision_switches_test",
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      model_name="hy4-tiny",
      per_device_batch_size=1,
      max_target_length=32,
      max_prefill_predict_length=32,
      dtype="bfloat16",
      weight_dtype="float32",
      attention="dot_product",
      sparse_matmul=False,
      megablox=False,
      **kw,
  )


def _build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    return model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))


def _modules(model, cls):
  return [("/".join(map(str, path)), m) for path, m in model.iter_modules() if isinstance(m, cls)]


class Hy4SwitchWiringTest(unittest.TestCase):
  """Each switch reaches the intended Hy4 modules; iHC and the indexer are left alone."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.off = _build(_config())
    cls.on = _build(_config(**SWITCHES))

  def test_rmsnorm_switches(self):
    norms = _modules(self.on, RMSNorm)
    paths = [p for p, _ in norms]
    # Every RMSNorm on the Hy4 path: layer pre/post norms, MLA q/kv latent norms, decoder final norm, MTP norms.
    for name in (
        "pre_self_attention_layer_norm",
        "post_self_attention_layer_norm",
        "self_attention/q_norm",
        "self_attention/kv_norm",
        "decoder_norm",
        "embedding_norm",
        "hidden_state_norm",
        "final_norm",
    ):
      self.assertTrue(any(p.endswith(name) for p in paths), (name, paths))
    for p, m in norms:
      self.assertTrue(m.single_rounding and m.scale_grad_fp32, p)
    # The MTP block's layer also has its own pre/post norms and MLA latent norms.
    self.assertTrue(any("mtp" in p and p.endswith("self_attention/q_norm") for p in paths), paths)
    for p, m in _modules(self.off, RMSNorm):
      self.assertFalse(m.single_rounding or m.scale_grad_fp32, p)

  def test_dense_fp32_accumulation(self):
    dense = _modules(self.on, linears.DenseGeneral)
    want = ("wq_a", "wq_b", "wkv_a", "wkv_b", "out", "gate_proj", "wi_0", "wi_1", "wo", "mtp_1_projection")
    covered = [p for p, m in dense if m.fp32_accumulation]
    for name in want:
      self.assertTrue(any(p.endswith("/" + name) and "indexer" not in p for p in covered), (name, covered))
    # shared experts and the dense first layer
    self.assertTrue(any("shared_experts" in p for p in covered), covered)
    for p, m in dense:
      if "indexer" in p or p.endswith("logits_dense"):
        self.assertFalse(m.fp32_accumulation, p)  # indexer (incl. weights_proj) and the head are not covered
      else:
        self.assertTrue(m.fp32_accumulation, p)
    self.assertTrue(any("indexer" in p for p, _ in dense))
    for p, m in _modules(self.off, linears.DenseGeneral):
      self.assertFalse(m.fp32_accumulation or m.kernel_grad_fp32, p)

  def test_head_and_gate(self):
    head = [m for p, m in _modules(self.on, linears.DenseGeneral) if p.endswith("logits_dense")]
    self.assertEqual(len(head), 1)
    self.assertTrue(head[0].kernel_grad_fp32)
    gates = _modules(self.on, moe.GateLogit)
    self.assertTrue(gates)
    for p, g in gates:
      self.assertTrue(g.bf16_round_kernel, p)
      self.assertEqual(g.matmul_precision, "highest", p)  # hy4-tiny router_matmul_precision
    for p, g in _modules(self.off, moe.GateLogit):
      self.assertFalse(g.bf16_round_kernel, p)

  def test_mlp_blocks_read_config(self):
    blocks = _modules(self.on, linears.MlpBlock)
    self.assertTrue(any("shared_experts" in p for p, _ in blocks))
    self.assertTrue(any(p.endswith("/mlp") for p, _ in blocks))  # the dense first layer
    for p, b in blocks:
      self.assertTrue(b.config.mlp_activation_round_per_op, p)


class Hy4ForwardBackwardTest(unittest.TestCase):
  """Forward + backward through hy4-tiny (deepseek + MLA + indexer + MoE + iHC + MTP), switches off vs on."""

  def _loss_and_grads(self, cfg):
    model = _build(cfg)
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, (1, cfg.max_target_length)), jnp.int32)
    pos = jnp.arange(cfg.max_target_length, dtype=jnp.int32)[None, :]
    seg = jnp.ones_like(tokens)

    def loss_fn(m):
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        out = m(tokens, pos, decoder_segment_ids=seg, enable_dropout=False, model_mode="train",
                decoder_target_tokens=tokens, decoder_target_mask=seg)
      logits = out[0] if isinstance(out, tuple) else out
      logp = jax.nn.log_softmax(logits.astype(F32), axis=-1)
      return -jnp.mean(jnp.take_along_axis(logp, tokens[..., None], axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    flat = {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(grads)}
    return float(loss), flat

  def test_switches_on_runs_and_changes_numerics(self):
    loss_off, g_off = self._loss_and_grads(_config())
    loss_on, g_on = self._loss_and_grads(_config(**SWITCHES))
    self.assertTrue(np.isfinite(loss_off) and np.isfinite(loss_on))
    self.assertEqual(sorted(g_off), sorted(g_on))
    for k, v in g_on.items():
      self.assertTrue(np.all(np.isfinite(v)), k)
      self.assertEqual(v.dtype, g_off[k].dtype, k)
    # Same weights; the switches change the rounding, so the gradients move but stay close.
    diff = [k for k in g_on if g_on[k].size and not np.array_equal(g_on[k], g_off[k])]
    self.assertTrue(diff)
    self.assertAlmostEqual(loss_on, loss_off, delta=1e-2 * abs(loss_off))

  def test_explicit_off_equals_default(self):
    loss_a, g_a = self._loss_and_grads(_config())
    loss_b, g_b = self._loss_and_grads(_config(**{k: False for k in SWITCHES}))
    self.assertEqual(loss_a, loss_b)
    for k in g_a:
      np.testing.assert_array_equal(g_a[k], g_b[k], err_msg=k)


class Hy4RoutedActivationTest(unittest.TestCase):
  """mlp_activation_round_per_op on the Hy4 routed path keeps the SwiGLU clamp (mlp_activations_limit 10)."""

  def test_per_op_keeps_clamp(self):
    rng = np.random.default_rng(3)
    g = jnp.asarray(rng.standard_normal((64, 96)) * 8, BF16)  # many values beyond +-10
    u = jnp.asarray(rng.standard_normal((64, 96)) * 8, BF16)
    self.assertGreater(float(jnp.max(jnp.abs(g.astype(F32)))), 10.0)
    cfg = types.SimpleNamespace(
        mlp_activation_round_per_op=True,
        activations_in_float32=False,
        decoder_block=DecoderBlockType.DEEPSEEK,
        mlp_activations_limit=10.0,
        mlp_activations_limit_inclusive_grad=True,
    )
    fake = types.SimpleNamespace(config=cfg, activation_fn=jax.nn.silu, dtype=BF16)
    out = moe.RoutedMoE.apply_ffn_activation(fake, g, u)
    gc = jnp.minimum(g.astype(F32), 10.0)
    uc = jnp.clip(u.astype(F32), -10.0, 10.0)
    want = (jax.nn.silu(gc).astype(BF16).astype(F32) * uc).astype(BF16)
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(want, np.float32))


class Hy4GateRoundingTest(unittest.TestCase):
  """routed_gate_bf16_rounding with hy4's float32_gate_logits + router_matmul_precision=highest: the router logits
  are exactly bf16 activation x bf16-rounded weight, accumulated in float32."""

  def test_gate_logits_match_bf16_operands(self):
    cfg = _config(routed_gate_bf16_rounding=True)
    model = _build(cfg)
    gate = [m for _, m in _modules(model, moe.GateLogit)][0]
    kernel = np.asarray(gate.kernel.get_value(), np.float32)
    if kernel.ndim == 3:  # scanned: [in, layers, out] or [layers, ...]; take one layer
      gate_kernel = kernel[:, 0, :] if kernel.shape[0] == cfg.emb_dim else kernel[0]
    else:
      gate_kernel = kernel
    single = moe.GateLogit(
        in_features_shape=cfg.emb_dim,
        out_features_shape=cfg.num_experts,
        mesh=Mesh(np.array(jax.devices()[:1]), ("data",)),
        model_name=cfg.model_name,
        rngs=nnx.Rngs(params=0),
        dtype=F32,
        weight_dtype=F32,
        score_func="",
        matmul_precision=gate.matmul_precision,
        bf16_round_kernel=gate.bf16_round_kernel,
    )
    single.kernel.set_value(jnp.asarray(gate_kernel))
    x = jnp.asarray(np.random.default_rng(4).standard_normal((2, 8, cfg.emb_dim)), BF16)
    out, _ = single(x.astype(F32))
    k_bf16 = np.asarray(jnp.asarray(gate_kernel, BF16).astype(F32), np.float64)
    ref = np.einsum("bsd,de->bse", np.asarray(x.astype(F32), np.float64), k_bf16)
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
