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
"""moe_mcore_prob_before_fc2: the routed experts' order of Megatron-Core (T1, PyTorch eager, alltoall dispatcher).

GPU: act = round_bf16(silu(gate)); h = round_bf16(act * up); h = round_bf16(h * prob_fp32) (type promotion to float32);
y = fc2(h) in bf16; the top-k y of a token are scatter_add_-ed into a bf16 zero tensor (one rounding per add).
MaxText (off): the routing weight multiplies the wo outputs inside the unpermute weight sum.

- The two helpers against the GPU order written out in numpy (ml_dtypes bf16), bit for bit, forward and backward.
- The routed MoE block on CPU (ragged_dot, no megablox), off and on, each against a dense per-token reference in its
  own order (MaxText's / the GPU's) with the block's rounding points: output and every gradient (input, wi_0, wi_1,
  wo, gate kernel) bit for bit.
"""

import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import ml_dtypes
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

BF = ml_dtypes.bfloat16


def _bf(v):
  return np.asarray(v, np.float32).astype(BF).astype(np.float32)


class HelpersTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    self.t, self.k, self.f, self.e = 16, 2, 24, 8
    self.inter = jnp.asarray(_bf(rng.standard_normal((self.t * self.k, self.f))), jnp.bfloat16)
    self.w = jnp.asarray(rng.uniform(0.05, 0.6, (1, self.t, self.k)).astype(np.float32))
    self.order = jnp.asarray(rng.permutation(self.t * self.k).astype(np.int32))
    self.y = jnp.asarray(_bf(rng.standard_normal((self.t, self.k, self.e))), jnp.bfloat16)

  def test_prob_before_fc2_forward_backward(self):
    got, vjp = jax.vjp(lambda h, w: moe.mcore_prob_before_fc2(h, w, self.order), self.inter, self.w)
    wf = np.ravel(np.asarray(self.w))[np.asarray(self.order)]
    want = _bf(np.asarray(self.inter, np.float32) * wf[:, None])
    np.testing.assert_array_equal(np.asarray(got, np.float32), want)
    g = jnp.asarray(_bf(np.random.default_rng(1).standard_normal(got.shape)), jnp.bfloat16)
    dh, dw = vjp(g)
    # PyTorch: d(h * p)/dh = g * p (float32, rounded to bf16), d/dp = sum_f g * h (float32), scattered back to slots
    np.testing.assert_array_equal(np.asarray(dh, np.float32), _bf(np.asarray(g, np.float32) * wf[:, None]))
    dws = (np.asarray(g, np.float32) * np.asarray(self.inter, np.float32)).sum(-1)
    want_dw = np.zeros(self.t * self.k, np.float32)
    want_dw[np.asarray(self.order)] = dws
    np.testing.assert_allclose(np.ravel(np.asarray(dw)), want_dw, rtol=1e-6, atol=1e-7)

  def test_sequential_sum(self):
    got = moe.mcore_sequential_sum(self.y, jnp.bfloat16)
    y = np.asarray(self.y, np.float32)
    acc = y[:, 0]
    for k in range(1, self.k):
      acc = _bf(acc + y[:, k])
    np.testing.assert_array_equal(np.asarray(got, np.float32), acc)
    self.assertEqual(got.dtype, jnp.bfloat16)


def _config(**kw):
  return pyconfig.initialize(
      [None, get_test_config_path()],
      run_name="moe_mcore_prob_before_fc2_test",
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
      **kw,
  )


class BlockTest(unittest.TestCase):

  def _build(self, cfg):
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
    return model, mesh

  def _run(self, model, mesh, cfg, x):
    gd, st, rest = nnx.split(model, nnx.Param, ...)

    def loss(st_, x_):
      out = nnx.merge(gd, st_, rest)(x_)[0]
      return jnp.sum(out.astype(jnp.float32) * self.cot), out

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      (_, out), (gst, gx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(st, x)
    return out, gst, gx

  def _reference(self, model, cfg, x, before):
    """Dense per-token routed experts with the same routing, in float32 with a BF16 rounding (reduce_precision, whose
    transpose rounds the cotangent too) at every point where the block holds a BF16 tensor, forward and backward.

    before=True: the GPU order (h * prob rounded before fc2, top-k outputs added one by one in BF16); before=False:
    MaxText's order (weights on the wo outputs, float32 weighted sum). Backward rounding points of the block, found
    by bisection: the input gradient of each of the two up projections (wi_0, wi_1 are separate GMMs) is rounded per
    (token, k) row, the k rows are summed in BF16, the router's input gradient is rounded on its own, and the two
    are added in BF16. (Megatron-Core fuses wi_0 / wi_1 into one fc1 GEMM, so its per-row input gradient is rounded
    once, not twice: a difference that does not depend on this switch.)
    """
    fi = jnp.finfo(jnp.bfloat16)
    rnd = lambda v: jax.lax.reduce_precision(v, fi.nexp, fi.nmant)  # noqa: E731

    def f(params, xf):
      wi0, wi1, wo, gk = params
      logits = jnp.einsum("bse,ex->bsx", rnd(xf), gk)
      w, sel = model.get_topk(logits, None, None)  # pylint: disable=protected-access
      xe = rnd(xf)
      ys, out = [], 0.0
      for k in range(cfg.num_experts_per_tok):
        e = sel[..., k]
        xk = rnd(xe)
        g = rnd(jnp.einsum("bse,bsef->bsf", rnd(xk), wi0[e]))
        u = rnd(jnp.einsum("bse,bsef->bsf", rnd(xk), wi1[e]))
        h = rnd(rnd(jax.nn.silu(g)) * u)
        if before:
          h = rnd(h * w[..., k : k + 1])
        y = rnd(jnp.einsum("bsf,bsfe->bse", h, wo[e]))
        ys.append(y)
        out = out + y * w[..., k : k + 1]
      if not before:
        return rnd(out)
      acc = rnd(ys[0])
      for y in ys[1:]:
        acc = rnd(acc + y)
      return acc

    params = tuple(
        p.astype(jnp.float32) for p in (model.wi_0[...], model.wi_1[...], model.wo[...], model.gate.kernel[...])
    )
    out, vjp = jax.vjp(f, params, x.astype(jnp.float32))
    gp, gx = vjp(rnd(self.cot.astype(jnp.float32)))
    return out.astype(jnp.bfloat16), [g.astype(jnp.bfloat16) for g in gp], gx.astype(jnp.bfloat16)

  def setUp(self):
    self.cfg_off = _config()
    self.cfg_on = _config(moe_mcore_prob_before_fc2=True)
    b = int(self.cfg_off.per_device_batch_size) * jax.device_count()
    rng = np.random.default_rng(2)
    self.x = jnp.asarray(rng.standard_normal((b, 8, 32)), jnp.bfloat16)
    self.cot = jnp.asarray(_bf(rng.standard_normal((b, 8, 32))))

  def test_on_and_off_match_their_references_bit_for_bit(self):
    outs = {}
    for name, cfg, before in (("off", self.cfg_off, False), ("on", self.cfg_on, True)):
      model, mesh = self._build(cfg)
      out, gst, gx = self._run(model, mesh, cfg, self.x)
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
        ref, gp, gx_ref = self._reference(model, cfg, self.x, before)
      eq = lambda a, b: np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))  # noqa: E731
      eq(out, ref)
      eq(gx, gx_ref)
      pure = nnx.to_pure_dict(gst)
      for key, g_ref in zip(("wi_0", "wi_1", "wo"), gp[:3]):
        eq(pure[key], g_ref)
      eq(pure["gate"]["kernel"], gp[3])
      outs[name] = np.asarray(out, np.float64)
    # negative control: the switch changes the result (other rounding points), by about one BF16 rounding
    rel = np.linalg.norm(outs["on"] - outs["off"]) / np.linalg.norm(outs["off"])
    self.assertGreater(rel, 0.0)
    self.assertLess(rel, 2e-2)

  def test_default_off(self):
    self.assertFalse(self.cfg_off.moe_mcore_prob_before_fc2)


if __name__ == "__main__":
  unittest.main()
