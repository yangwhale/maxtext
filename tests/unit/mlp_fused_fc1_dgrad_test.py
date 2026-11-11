# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mlp_fused_fc1_dgrad: the SwiGLU up projections' input gradient as Megatron-Core's fused linear_fc1 (one GEMM for
gate | up): dx = [g0 | g1] [w0 | w1]^T accumulated in float32, rounded to bf16 once.

Reference: the MLP written in float32 with a bf16 rounding (reduce_precision; its transpose rounds the cotangent too)
at every bf16 tensor. Without a rounding on x before the two projections, autodiff adds the two input gradients in
float32 and the final cast rounds once (the fused form); with a rounding before each projection, each is rounded on
its own (the unfused form, what the block computes when the key is off).

- MlpBlock (dense layers, shared experts): off equals the unfused reference and on the fused one, bit for bit; the
  forward and the kernel gradients are the same bits off and on.
- The routed experts (sparse_matmul, CPU ragged_dot): the same, on the routed MoE block.
"""

import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import linears, moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

BF16 = jnp.bfloat16
_FI = jnp.finfo(BF16)


def rnd(v):
  return jax.lax.reduce_precision(v, _FI.nexp, _FI.nmant)


def _eq(a, b, what=""):
  np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32), err_msg=what)


class MlpBlockTest(unittest.TestCase):

  def _block(self, fused):
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=16,
        fused_mlp=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        mlp_activations=["silu", "linear"],
        mlp_activation_round_per_op=True,
        mlp_fused_fc1_dgrad=fused,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    blk = linears.MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=32,
        intermediate_dim=48,
        dtype=BF16,
        weight_dtype=BF16,
        activations=["silu", "linear"],
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    return blk, cfg, mesh

  def _grads(self, blk, cfg, mesh, x, cot):
    gd, params, rest = nnx.split(blk, nnx.Param, ...)

    def loss(p, x_):
      out = nnx.merge(gd, p, rest)(x_, deterministic=True)
      return jnp.sum(out.astype(jnp.float32) * cot), out

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      (_, out), (gp, gx) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, x)
    return out, nnx.to_pure_dict(gp), gx

  @staticmethod
  def _reference(k0, k1, wo, x, cot, fused):
    def f(ks, xf):
      a, b, c = ks
      g = rnd(jnp.einsum("bse,em->bsm", xf if fused else rnd(xf), a))
      u = rnd(jnp.einsum("bse,em->bsm", xf if fused else rnd(xf), b))
      h = rnd(rnd(jax.nn.silu(g)) * u)
      return rnd(jnp.einsum("bsm,me->bse", h, c))

    ks = tuple(k.astype(jnp.float32) for k in (k0, k1, wo))
    out, vjp = jax.vjp(f, ks, x.astype(jnp.float32))
    gk, gx = vjp(rnd(cot))
    return out.astype(BF16), [g.astype(BF16) for g in gk], gx.astype(BF16)

  def test_off_unfused_on_fused(self):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((2, 16, 32)), BF16)
    cot = jnp.asarray(rng.standard_normal((2, 16, 32)).astype(np.float32)).astype(BF16).astype(jnp.float32)
    res = {}
    for fused in (False, True):
      blk, cfg, mesh = self._block(fused)
      out, gp, gx = self._grads(blk, cfg, mesh, x, cot)
      k0, k1, wo = (blk.wi_0.kernel[...], blk.wi_1.kernel[...], blk.wo.kernel[...])
      ref_out, gk, gx_ref = self._reference(k0, k1, wo, x, cot, fused)
      _eq(out, ref_out, "forward")
      _eq(gx, gx_ref, "dx")
      for name, g in zip(("wi_0", "wi_1", "wo"), gk):
        _eq(gp[name]["kernel"], g, name)
      res[fused] = (out, gp, gx)
    _eq(res[False][0], res[True][0], "forward unchanged")
    for name in ("wi_0", "wi_1", "wo"):
      _eq(res[False][1][name]["kernel"], res[True][1][name]["kernel"], name + " unchanged")
    self.assertFalse(np.array_equal(np.asarray(res[False][2], np.float32), np.asarray(res[True][2], np.float32)))


class RoutedTest(unittest.TestCase):

  def _cfg(self, fused):
    return pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="mlp_fused_fc1_dgrad_test",
        enable_checkpointing=False,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        max_target_length=8,
        per_device_batch_size=1,
        num_experts=4,
        num_experts_per_tok=2,
        base_emb_dim=32,
        base_mlp_dim=64,
        base_moe_mlp_dim=64,
        mlp_activations=["silu", "linear"],
        mlp_activation_round_per_op=True,
        sparse_matmul=True,
        megablox=False,
        float32_gate_logits=True,
        ici_fsdp_parallelism=-1,
        mlp_fused_fc1_dgrad=fused,
    )

  def test_off_unfused_on_fused(self):
    cfg0 = self._cfg(False)
    b = int(cfg0.per_device_batch_size) * jax.device_count()
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((b, 8, 32)), BF16)
    cot = jnp.asarray(rng.standard_normal((b, 8, 32)).astype(np.float32)).astype(BF16).astype(jnp.float32)
    res = {}
    for fused in (False, True):
      cfg = self._cfg(fused)
      mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
      model = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.base_moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          rngs=nnx.Rngs(params=0),
      )
      gd, params, rest = nnx.split(model, nnx.Param, ...)

      def loss(p, x_, gd=gd, rest=rest):
        out = nnx.merge(gd, p, rest)(x_)[0]
        return jnp.sum(out.astype(jnp.float32) * cot), out

      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        (_, out), (gp, gx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, x)
        ref = self._reference(model, cfg, x, cot, fused)
      _eq(out, ref[0], "forward")
      _eq(gx, ref[2], "dx")
      pure = nnx.to_pure_dict(gp)
      for name, g in zip(("wi_0", "wi_1", "wo"), ref[1][:3]):
        _eq(pure[name], g, name)
      _eq(pure["gate"]["kernel"], ref[1][3], "gate")
      res[fused] = (out, gx)
    _eq(res[False][0], res[True][0], "forward unchanged")
    self.assertFalse(np.array_equal(np.asarray(res[False][1], np.float32), np.asarray(res[True][1], np.float32)))

  @staticmethod
  def _reference(model, cfg, x, cot, fused):
    """Rounding points of the routed block's backward (moe_mcore_prob_before_fc2_test): the router's and the experts'
    input gradients rounded apart, the k rows added in bf16, and each (token, k) row's input gradient rounded once
    (fused) or once per projection (unfused)."""

    def f(params, xf):
      wi0, wi1, wo, gk = params
      logits = jnp.einsum("bse,ex->bsx", rnd(xf), gk)
      w, sel = model.get_topk(logits, None, None)  # pylint: disable=protected-access
      xe = rnd(xf)
      out = 0.0
      for k in range(cfg.num_experts_per_tok):
        e = sel[..., k]
        xk = rnd(xe)
        g = rnd(jnp.einsum("bse,bsef->bsf", xk if fused else rnd(xk), wi0[e]))
        u = rnd(jnp.einsum("bse,bsef->bsf", xk if fused else rnd(xk), wi1[e]))
        h = rnd(rnd(jax.nn.silu(g)) * u)
        out = out + rnd(jnp.einsum("bsf,bsfe->bse", h, wo[e])) * w[..., k : k + 1]
      return rnd(out)

    params = tuple(
        p.astype(jnp.float32) for p in (model.wi_0[...], model.wi_1[...], model.wo[...], model.gate.kernel[...])
    )
    out, vjp = jax.vjp(f, params, x.astype(jnp.float32))
    gp, gx = vjp(rnd(cot))
    return out.astype(BF16), [g.astype(BF16) for g in gp], gx.astype(BF16)


if __name__ == "__main__":
  unittest.main()
