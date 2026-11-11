# Copyright 2026 Google LLC
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
"""Tests for the fused iHC Pallas kernels (kernels/ihc), run in interpret mode off TPU.

Accuracy is judged against a float64 evaluation of the iHC formulas: the kernel must be at least as close to it as
the XLA float32 path (HIGHEST gate projection) it replaces.
"""

import types
import unittest

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import HyperConnectionType
from maxtext.configs import pyconfig
from maxtext.kernels import ihc as ihc_kernel
from maxtext.kernels.ihc import ihc_kernels
from maxtext.layers import mhc
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np

EPS, HC_EPS, MAG = 1e-5, 1e-6, 2.0
CFG = ihc_kernel.IhcKernelConfig(
    rms_epsilon=EPS, hc_eps=HC_EPS, post_magnitude=MAG, block_size=32, bwd_block_size=16, interpret=True
)


def _inputs(b=2, s=32, k=4, d=256, seed=0, alpha_std=0.05):
  ks = jax.random.split(jax.random.PRNGKey(seed), 8)
  x = jax.random.normal(ks[0], (b, s, k, d), jnp.float32).astype(jnp.bfloat16)
  alpha = alpha_std * jax.random.normal(ks[1], (k * d, 2 * k), jnp.float32)
  scale = 1.0 + 0.5 * jax.random.normal(ks[2], (2 * k,), jnp.float32)
  beta = 0.5 * jax.random.normal(ks[3], (2 * k,), jnp.float32)
  d_li = jax.random.normal(ks[4], (b, s, d), jnp.float32).astype(jnp.bfloat16)
  d_x = jax.random.normal(ks[5], (b, s, k, d), jnp.float32).astype(jnp.bfloat16)
  d_post = jax.random.normal(ks[6], (b, s, k), jnp.float32)
  lo = jax.random.normal(ks[7], (b, s, d), jnp.float32).astype(jnp.bfloat16)
  return x, alpha, scale, beta, d_li, d_x, d_post, lo


def _pre_reference(x, alpha, scale, beta, dtype, precision=jax.lax.Precision.HIGHEST):
  """The iHC gates and stream sum as in IdentityHyperConnections (XLA path), computed in `dtype`, outputs uncast."""
  b, s, k, d = x.shape
  flat = x.reshape(b, s, k * d).astype(dtype)
  inv = jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + EPS)
  mixes = jnp.einsum("bsm,mn->bsn", flat, alpha.astype(dtype), precision=precision) * inv
  z = mixes * scale.astype(dtype) + beta.astype(dtype)
  pre = jax.nn.sigmoid(z[..., :k]) + HC_EPS
  post = MAG * jax.nn.sigmoid(z[..., k:]) + HC_EPS
  li = jnp.sum(pre[..., None] * x.astype(dtype), axis=2)
  return li, x.astype(dtype), post


def _pre_vjp(fn, primals, cts):
  out, vjp = jax.vjp(fn, *primals)
  return out, vjp(cts)


def _f64(t):
  return np.asarray(jnp.asarray(t, jnp.float32), np.float64)


def _err(a, ref):
  return np.max(np.abs(_f64(a) - ref))


class SplitTest(unittest.TestCase):

  def test_split3_is_exact(self):
    a = jax.random.normal(jax.random.PRNGKey(1), (4096,), jnp.float32) * 0.01
    hi, mid, lo = ihc_kernels._split3(a)
    with jax.enable_x64(True):
      total = hi.astype(jnp.float64) + mid.astype(jnp.float64) + lo.astype(jnp.float64)
      np.testing.assert_array_equal(np.asarray(total), np.asarray(a, np.float64))
    # hi and mid are truncations of the bit pattern, so they carry no low bits that a float32 -> bfloat16 cast could
    # round; a round-trip-based split would be folded away by XLA's excess-precision rewrite on TPU.
    for piece in (hi, mid):
      bits = jax.lax.bitcast_convert_type(piece.astype(jnp.float32), jnp.int32)
      np.testing.assert_array_equal(np.asarray(bits) & 0xFFFF, 0)
    # Negative control: two pieces are not enough for float32.
    two = np.asarray(hi, np.float64) + np.asarray(mid, np.float64)
    self.assertGreater(np.mean(two != np.asarray(a, np.float64)), 0.5)


class PreKernelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.x, self.alpha, self.scale, self.beta, self.d_li, self.d_x, self.d_post, _ = _inputs()
    primals = (self.x, self.alpha, self.scale, self.beta)
    cts = (self.d_li, self.d_x, self.d_post)
    self.kernel = _pre_vjp(lambda *p: ihc_kernel.ihc_pre(CFG, *p), primals, cts)
    self.xla = _pre_vjp(lambda *p: _pre_reference(*p, jnp.float32), primals, (self.d_li.astype(jnp.float32), self.d_x.astype(jnp.float32), self.d_post))
    # x enters the float64 reference as float64: with a bfloat16 x, JAX would round the cotangent of each of its three
    # uses to bfloat16 and add them in bfloat16, and the reference d_x would carry that cancellation error.
    with jax.enable_x64(True):
      out, grads = _pre_vjp(
          lambda *p: _pre_reference(*p, jnp.float64),
          (self.x.astype(jnp.float64), self.alpha.astype(jnp.float64), self.scale.astype(jnp.float64), self.beta.astype(jnp.float64)),
          (self.d_li.astype(jnp.float64), self.d_x.astype(jnp.float64), self.d_post.astype(jnp.float64)),
      )
      self.ref = (jax.tree.map(np.asarray, out), jax.tree.map(np.asarray, grads))

  def test_x_passthrough(self):
    np.testing.assert_array_equal(np.asarray(self.kernel[0][1]), np.asarray(self.x))

  def test_post_gates_float32_accurate(self):
    k_err = _err(self.kernel[0][2], self.ref[0][2])
    x_err = _err(self.xla[0][2], self.ref[0][2])
    self.assertLess(k_err, 1e-6)
    self.assertLessEqual(k_err, 2 * x_err + 1e-7)

  def test_layer_input_within_one_bf16_ulp(self):
    ref = self.ref[0][0]
    got = _f64(self.kernel[0][0])
    self.assertEqual(self.kernel[0][0].dtype, jnp.bfloat16)
    np.testing.assert_array_less(np.abs(got - ref), np.abs(ref) * 2.0**-8 + 1e-6)

  @parameterized.named_parameters(("alpha", 1), ("scale", 2), ("beta", 3))
  def test_param_grads_float32_accurate(self, idx):
    ref = self.ref[1][idx]
    k_err = _err(self.kernel[1][idx], ref)
    x_err = _err(self.xla[1][idx], ref)
    scale = np.max(np.abs(ref))
    self.assertLess(k_err / scale, 1e-5)
    self.assertLessEqual(k_err, 2 * x_err + 1e-6 * scale)

  def test_x_grad_within_one_bf16_rounding(self):
    ref = self.ref[1][0]
    got = _f64(self.kernel[1][0])
    self.assertEqual(self.kernel[1][0].dtype, jnp.bfloat16)
    np.testing.assert_array_less(np.abs(got - ref), np.abs(ref) * 2.0**-8 + 1e-5)

  def test_x_grad_includes_each_path(self):
    """Dropping any of the three cotangent paths into d_x (residual passthrough, stream sum, gates) moves d_x far
    outside the bf16 tolerance, so the tolerance above does distinguish them."""
    x, alpha, scale, beta = self.x, self.alpha, self.scale, self.beta
    with jax.enable_x64(True):
      for cts in (
          (self.d_li, jnp.zeros_like(self.d_x), self.d_post),
          (jnp.zeros_like(self.d_li), self.d_x, self.d_post),
          (self.d_li, self.d_x, jnp.zeros_like(self.d_post)),
      ):
        _, grads = _pre_vjp(
            lambda *p: _pre_reference(*p, jnp.float64),
            (x.astype(jnp.float64), alpha.astype(jnp.float64), scale.astype(jnp.float64), beta.astype(jnp.float64)),
            tuple(c.astype(jnp.float64) for c in cts),
        )
        ref = np.asarray(grads[0])
        self.assertTrue(np.any(np.abs(_f64(self.kernel[1][0]) - ref) > np.abs(ref) * 2.0**-8 + 1e-5))


class PostKernelTest(unittest.TestCase):

  def test_forward_and_vjp(self):
    x, _, _, _, _, d_x, d_post, lo = _inputs(seed=3)
    post = 0.1 + MAG * jax.nn.sigmoid(d_post)
    out, (d_lo, g_x, g_post) = _pre_vjp(lambda *p: ihc_kernel.ihc_post(CFG, *p), (lo, x, post), d_x)
    with jax.enable_x64(True):
      ref_fn = lambda lo, x, post: x.astype(jnp.float64) + post[..., None] * lo.astype(jnp.float64)[:, :, None]
      ref, (r_lo, r_x, r_post) = _pre_vjp(ref_fn, (lo, x, post.astype(jnp.float64)), d_x.astype(jnp.float64))
      ref, r_lo, r_x, r_post = map(np.asarray, (ref, r_lo, r_x, r_post))
    np.testing.assert_array_less(np.abs(_f64(out) - ref), np.abs(ref) * 2.0**-8 + 1e-6)
    # r_lo is rounded to bfloat16 like d_lo (lo is bfloat16): one ulp apart at most.
    np.testing.assert_array_less(np.abs(_f64(d_lo) - _f64(r_lo)), np.abs(_f64(r_lo)) * 2.0**-7 + 1e-5)
    np.testing.assert_array_equal(np.asarray(g_x), np.asarray(d_x))
    np.testing.assert_allclose(_f64(g_post), r_post, rtol=1e-5, atol=1e-4)


class ModuleTest(parameterized.TestCase):
  """IdentityHyperConnections with use_ihc_pallas_kernel on vs off, through the mesh / shard_map path."""

  def _module(self, use_kernel, d=256, seq=64, **extra):
    config = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="test_ihc_kernel",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        per_device_batch_size=1,
        max_target_length=seq,
        max_prefill_predict_length=seq,
        attention="dot_product",
        attention_type="mla",
        override_model_config=True,
        base_emb_dim=d,
        mhc_expansion_rate=4,
        hc_variant="ihc",
        ihc_stream_sum_unrolled=True,
        use_ihc_pallas_kernel=use_kernel,
        ihc_pallas_block_size=32,
        ihc_pallas_bwd_block_size=16,
        dtype="bfloat16",
        decoder_block="deepseek",
        **{"engram_layers": [], **extra},
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
    return config, mesh

  def _value_and_grads(self, use_kernel):
    config, mesh = self._module(use_kernel)
    x = jax.random.normal(jax.random.PRNGKey(5), (jax.device_count(), 64, 4, 256), jnp.float32).astype(jnp.bfloat16)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      module = mhc.make_hyper_connections(config, 256, mesh, nnx.Rngs(params=0))
      graphdef, params, rest = nnx.split(module, nnx.Param, ...)

      # A linear branch and loss give both paths the same cotangent; with e.g. sum(out**2) the cotangent would carry
      # each path's own bfloat16 rounding of out, which the cancelling sums in the gate-scale gradients amplify.
      w = jax.random.normal(jax.random.PRNGKey(6), x.shape, jnp.float32)

      def loss(params, x):
        m = nnx.merge(graphdef, params, rest)
        out, _ = m(lambda v: v, lambda inputs: 0.5 * inputs, x=x, mhc_type=HyperConnectionType.MLP_DENSE)
        return jnp.sum(out.astype(jnp.float32) * w)

      fn = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
      jaxpr = str(jax.make_jaxpr(fn)(params, x))
      val, (g_params, g_x) = fn(params, x)
    return jaxpr, val, g_params, g_x

  def test_kernel_matches_xla_path(self):
    j0, v0, p0, x0 = self._value_and_grads(False)
    j1, v1, p1, x1 = self._value_and_grads(True)
    self.assertNotIn("pallas_call", j0)
    self.assertIn("pallas_call", j1)
    self.assertIn("shard_map", j1)
    # Both paths round layer_input and the branch cotangent to bfloat16; where their float32 values differ in the last
    # bits a rounding can flip, so module-level sums agree to ~1e-3, not bitwise. Kernel accuracy for identical inputs
    # is checked against float64 in PreKernelTest / PostKernelTest.
    np.testing.assert_allclose(v1, v0, rtol=1e-2)
    # The XLA path rounds the cotangent of each use of x to bfloat16 and adds them in bfloat16, so where its terms
    # (|.| up to ~8) cancel it is off by a few ulps of the terms; the kernel adds in float32 and rounds once (its
    # accuracy is checked against float64 in PreKernelTest).
    np.testing.assert_allclose(np.asarray(x1, np.float32), np.asarray(x0, np.float32), rtol=2e-2, atol=0.1)
    for a, b in zip(jax.tree.leaves(p0), jax.tree.leaves(p1)):
      np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-2, atol=1e-2 * np.max(np.abs(np.asarray(a))))

  @parameterized.named_parameters(("kernel", True), ("xla", False))
  def test_flat_streams_match_4d(self, use_kernel):
    """Streams carried flat, (batch, seq, k * emb), give the 4D result reshaped, values and gradients."""
    config, mesh = self._module(use_kernel)
    x = jax.random.normal(jax.random.PRNGKey(5), (jax.device_count(), 64, 4, 256), jnp.float32).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.PRNGKey(6), x.shape, jnp.float32)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      module = mhc.make_hyper_connections(config, 256, mesh, nnx.Rngs(params=0))
      graphdef, params, rest = nnx.split(module, nnx.Param, ...)

      def loss(params, x):
        m = nnx.merge(graphdef, params, rest)
        out, _ = m(lambda v: v, lambda inputs: 0.5 * inputs, x=x, mhc_type=HyperConnectionType.MLP_DENSE)
        return jnp.sum(out.astype(jnp.float32).reshape(w.shape) * w), out.shape

      (v4, s4), (p4, x4) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, x)
      (vf, sf), (pf, xf) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(params, x.reshape(x.shape[:2] + (-1,)))
    self.assertEqual(sf, x.shape[:2] + (4 * 256,))
    np.testing.assert_array_equal(vf, v4)
    np.testing.assert_array_equal(np.asarray(xf, np.float32).reshape(x.shape), np.asarray(x4, np.float32))
    for a, b in zip(jax.tree.leaves(p4), jax.tree.leaves(pf)):
      np.testing.assert_array_equal(np.asarray(b), np.asarray(a))

  @parameterized.named_parameters(("kernel", 64), ("xla_fallback", 8))
  def test_decoder_with_flat_streams_matches_4d(self, seq):
    """Whole decoder (dense + MoE layers, hc_head) with use_ihc_pallas_kernel on (flat streams) vs off (4D streams).

    The module tests above call IdentityHyperConnections directly; this one runs the layers around it, which see the
    flat (batch, seq, k * emb) carry (R055 E047 first failed there: a post-attention norm applied to the stream).
    seq=64 tiles into the kernel blocks; seq=8 does not, so the flat carry goes through the XLA fallback, where the
    math is the 4D path's and only reshapes differ.
    """
    from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel

    out = {}
    for use_kernel in (False, True):
      config, mesh = self._module(
          use_kernel, seq=seq, base_num_decoder_layers=2, first_num_dense_layers=1, num_experts=4,
          num_experts_per_tok=2, base_moe_mlp_dim=128, base_mlp_dim=256, vocab_size=512, sparse_matmul=False,
          megablox=False, capacity_factor=-1.0, mtp_num_layers=0, scan_layers=True, use_indexer=False,
          use_hc_head=True,
      )
      self.assertEqual(mhc.streams_flat(config), use_kernel)
      tokens = jax.random.randint(jax.random.PRNGKey(7), (config.global_batch_size_to_train_on, seq), 0, 512)
      pos = jnp.broadcast_to(jnp.arange(seq), tokens.shape)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        model = model_creation_utils.from_config(config, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
        self.assertTrue(model.decoder.use_hc_head)
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss(params):
          m = nnx.merge(graphdef, params, rest)
          logits = m(tokens, pos, decoder_segment_ids=jnp.ones_like(tokens), enable_dropout=False)
          return jnp.mean(logits.astype(jnp.float32) ** 2), logits

        fn = jax.jit(jax.value_and_grad(loss, has_aux=True))
        jaxpr = str(jax.make_jaxpr(fn)(params))
        (val, logits), grads = fn(params)
      out[use_kernel] = (jaxpr, val, logits, grads)
    self.assertNotIn("pallas_call", out[False][0])
    g0, g1 = jax.tree.leaves(out[False][3]), jax.tree.leaves(out[True][3])
    self.assertEqual(len(g0), len(g1))
    if seq == 8:
      # Fallback: same XLA math on a reshaped carry -- values and every gradient bitwise equal.
      self.assertNotIn("pallas_call", out[True][0])
      np.testing.assert_array_equal(out[True][1], out[False][1])
      np.testing.assert_array_equal(np.asarray(out[True][2], np.float32), np.asarray(out[False][2], np.float32))
      for a, b in zip(g0, g1):
        np.testing.assert_array_equal(np.asarray(b, np.float32), np.asarray(a, np.float32))
      return
    self.assertIn("pallas_call", out[True][0])
    # Same tolerances as test_kernel_matches_xla_path: the paths round in different places, not bitwise.
    np.testing.assert_allclose(out[True][1], out[False][1], rtol=2e-2)
    np.testing.assert_allclose(
        np.asarray(out[True][2], np.float32), np.asarray(out[False][2], np.float32), rtol=5e-2, atol=5e-2
    )
    floor = 1e-3 * np.sqrt(sum(np.sum(np.asarray(a, np.float64) ** 2) for a in g0))
    for a, b in zip(g0, g1):
      a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
      self.assertEqual(a.shape, b.shape)
      # Each gradient leaf agrees in norm to a few percent (a misrouted or zeroed gradient would not); leaves far below
      # the overall gradient scale get an absolute floor of 1e-3 of the total norm.
      self.assertLessEqual(np.linalg.norm(b - a), 5e-2 * np.linalg.norm(a) + floor)

  @parameterized.named_parameters(
      ("kernel", {}, True),
      ("kernel_off", {"use_ihc_pallas_kernel": False}, False),
      ("no_streams", {"mhc_expansion_rate": 1}, False),
      ("engram", {"engram_layers": [1]}, False),
      ("dropout", {"enable_dropout": True, "dropout_rate": 0.1}, False),
      ("dropout_rate_but_disabled", {"enable_dropout": False, "dropout_rate": 0.1}, True),
  )
  def test_streams_flat(self, extra, want):
    """streams_flat reads only these config fields (the real-config path is covered by the decoder test)."""
    fields = {"mhc_expansion_rate": 4, "use_ihc_pallas_kernel": True, "engram_layers": [], "enable_dropout": False,
              "dropout_rate": 0.0, **extra}
    self.assertEqual(mhc.streams_flat(types.SimpleNamespace(**fields)), want)

  def test_falls_back_to_xla_when_tokens_do_not_tile(self):
    """8 tokens per device (e.g. decode) do not tile into 32-row blocks: the XLA path runs instead of raising."""
    config, mesh = self._module(True, seq=8)
    x = jax.random.normal(jax.random.PRNGKey(5), (jax.device_count(), 8, 4, 256), jnp.float32).astype(jnp.bfloat16)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      module = mhc.make_hyper_connections(config, 256, mesh, nnx.Rngs(params=0))
      fn = lambda x: module(lambda v: v, lambda inputs: inputs, x=x, mhc_type=HyperConnectionType.MLP_DENSE)[0]
      jaxpr = str(jax.make_jaxpr(fn)(x))
      out = jax.jit(fn)(x)
    self.assertNotIn("pallas_call", jaxpr)
    self.assertEqual(out.shape, x.shape)

  @parameterized.named_parameters(
      ("six_streams", {"mhc_expansion_rate": 6}),
      ("default_precision", {"hc_matmul_precision": "default"}),
  )
  def test_config_rejects_unsupported(self, extra):
    kwargs = dict(
        run_name="test_ihc_kernel_bad",
        enable_checkpointing=False,
        model_name="deepseek-custom",
        override_model_config=True,
        mhc_expansion_rate=4,
        hc_variant="ihc",
        use_ihc_pallas_kernel=True,
    )
    kwargs.update(extra)
    with self.assertRaises(ValueError):
      pyconfig.initialize([None, get_test_config_path()], **kwargs)

  def test_kernel_rejects_more_than_max_streams(self):
    k = ihc_kernel.MAX_STREAMS + 1
    x = jnp.zeros((1, 32, k, 128), jnp.bfloat16)
    alpha = jnp.zeros((k * 128, 2 * k), jnp.float32)
    with self.assertRaisesRegex(ValueError, "streams"):
      ihc_kernel.ihc_pre(CFG, x, alpha, jnp.ones((2 * k,)), jnp.zeros((2 * k,)))

  def test_rejected_for_mhc(self):
    with self.assertRaises(ValueError):
      pyconfig.initialize(
          [None, get_test_config_path()],
          run_name="test_ihc_kernel_mhc",
          enable_checkpointing=False,
          model_name="deepseek-custom",
          override_model_config=True,
          mhc_expansion_rate=4,
          hc_variant="mhc",
          use_ihc_pallas_kernel=True,
      )

  def test_requires_bfloat16(self):
    x, alpha, scale, beta = _inputs()[:4]
    with self.assertRaises(ValueError):
      ihc_kernel.ihc_pre(CFG, x.astype(jnp.float32), alpha, scale, beta)


if __name__ == "__main__":
  unittest.main()
