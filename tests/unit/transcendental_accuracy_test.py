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

"""transcendental_accuracy_highest: float32 exp / log / sigmoid / silu at lax accuracy=HIGHEST.

- accurate_math helpers: off is the jax function bit for bit; on stays within 4 ULP of float64 (CPU here; the TPU
  default versions are 62-124 ULP off on v5p, JAX "Numerical accuracy");
- the switch reaches the sites: every exp / log / logistic in the traced iHC gates, router and MLP activation carries
  accuracy=HIGHEST when it is on, and none does when it is off.
"""

import unittest
from types import SimpleNamespace

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import linears, mhc
from maxtext.layers import multi_token_prediction as mtp
from maxtext.utils import accurate_math as am
from maxtext.utils import max_utils

# TestMHC only as a config / mesh helper; a module import keeps pytest from collecting its tests here.
from tests.unit import mhc_test


def _ulp_max(out, ref64):
  out = np.asarray(out)
  cr = ref64.astype(np.float32)
  sp = np.spacing(np.abs(cr)).astype(np.float64)
  return float(np.max(np.abs(out.astype(np.float64) - ref64) / sp))


def _accuracy_params(fn, *args):
  """accuracy params of every exp / log / logistic primitive in the jaxpr of fn(*args), recursively."""
  found = []

  def walk(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name in ("exp", "log", "logistic"):
        found.append((eqn.primitive.name, eqn.params.get("accuracy")))
      for p in eqn.params.values():
        if hasattr(p, "jaxpr"):
          walk(p.jaxpr if hasattr(p.jaxpr, "eqns") else p.jaxpr.jaxpr)

  walk(jax.make_jaxpr(fn)(*args).jaxpr)
  return found


class AccurateMathTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    self.x = rng.uniform(-20, 20, 1 << 14).astype(np.float32)
    self.pos = np.exp(rng.uniform(-30, 30, 1 << 14)).astype(np.float32)
    self.rows = rng.standard_normal((16, 512)).astype(np.float32) * 0.05

  def test_off_is_unchanged(self):
    x, p, r = jnp.asarray(self.x), jnp.asarray(self.pos), jnp.asarray(self.rows)
    np.testing.assert_array_equal(am.exp(x, False), jnp.exp(x))
    np.testing.assert_array_equal(am.log(p, False), jnp.log(p))
    np.testing.assert_array_equal(am.sigmoid(x, False), jax.nn.sigmoid(x))
    np.testing.assert_array_equal(am.silu(x, False), jax.nn.silu(x))
    np.testing.assert_array_equal(am.softmax(r, -1, False), jax.nn.softmax(r, axis=-1))
    np.testing.assert_array_equal(am.log_softmax(r, -1, False), jax.nn.log_softmax(r, axis=-1))
    self.assertIsNone(am.activation("silu", False))

  def test_on_within_4_ulp_of_float64(self):
    x64, p64, r64 = self.x.astype(np.float64), self.pos.astype(np.float64), self.rows.astype(np.float64)
    x, p, r = jnp.asarray(self.x), jnp.asarray(self.pos), jnp.asarray(self.rows)
    sig = 1.0 / (1.0 + np.exp(-x64))
    lse = np.log(np.exp(r64 - r64.max(-1, keepdims=True)).sum(-1, keepdims=True)) + r64.max(-1, keepdims=True)
    self.assertLessEqual(_ulp_max(am.exp(x, True), np.exp(x64)), 4)
    self.assertLessEqual(_ulp_max(am.log(p, True), np.log(p64)), 4)
    self.assertLessEqual(_ulp_max(am.sigmoid(x, True), sig), 4)
    self.assertLessEqual(_ulp_max(am.silu(x, True), x64 * sig), 4)
    self.assertLessEqual(_ulp_max(am.softmax(r, -1, True), np.exp(r64 - lse)), 4)
    self.assertLessEqual(_ulp_max(am.log_softmax(r, -1, True), r64 - lse), 4)

  def test_gradients_match_the_jax_functions(self):
    r = jnp.asarray(self.rows)
    w = jnp.asarray(np.random.default_rng(1).standard_normal(self.rows.shape).astype(np.float32))
    for f in (am.softmax, am.log_softmax):
      g_on = jax.grad(lambda v, f=f: jnp.sum(f(v, -1, True) * w))(r)
      g_off = jax.grad(lambda v, f=f: jnp.sum(f(v, -1, False) * w))(r)
      np.testing.assert_allclose(np.asarray(g_on), np.asarray(g_off), rtol=1e-5, atol=1e-6)

  def test_sigmoid_does_not_use_logistic(self):
    """On TPU v5p lax.logistic(accuracy=HIGHEST) gives the default 83-ULP result; the accurate sigmoid must avoid it."""
    x = jnp.asarray(self.x)
    for fn in (am.sigmoid, am.silu):
      self.assertNotIn("logistic", [n for n, _ in _accuracy_params(lambda v, fn=fn: fn(v, True), x)])

  def test_bf16_in_bf16_out(self):
    x = jnp.asarray(self.x, jnp.bfloat16)
    x64 = np.asarray(x, np.float64)
    sig = 1.0 / (1.0 + np.exp(-x64))
    for fn, ref in ((am.sigmoid, sig), (am.silu, x64 * sig)):
      y = fn(x, True)
      self.assertEqual(y.dtype, jnp.bfloat16)
      # computed in float32 and rounded once: the correctly rounded BF16 value, up to a tie
      want = ref.astype(np.float32).astype(jnp.bfloat16).astype(np.float64)
      got = np.asarray(y, np.float64)
      ulp = np.abs(want) * 2.0**-7 + 1e-40
      self.assertLessEqual(float(np.max(np.abs(got - want) / ulp)), 1.0)

  def test_gradient_of_the_accurate_sigmoid(self):
    x = jnp.asarray(self.x)
    for fn, ref in ((am.sigmoid, jax.nn.sigmoid), (am.silu, jax.nn.silu)):
      g = jax.grad(lambda v, fn=fn: jnp.sum(fn(v, True)))(x)
      g0 = jax.grad(lambda v, ref=ref: jnp.sum(ref(v)))(x)
      np.testing.assert_allclose(np.asarray(g), np.asarray(g0), rtol=1e-5, atol=1e-7)

  def test_backward_is_the_pytorch_formula(self):
    """sigmoid: g (1 - y) y from the stored output; silu: g s (1 + x (1 - s)) with s recomputed in float32 (PyTorch's
    CUDA sigmoid_backward / silu_backward), in the input dtype."""
    for dt in (jnp.float32, jnp.bfloat16):
      x = jnp.asarray(self.x, dt)
      g = jnp.asarray(np.random.default_rng(4).standard_normal(self.x.shape), dt)
      y, vjp = jax.vjp(lambda v: am.sigmoid(v, True), x)
      (dx,) = vjp(g)
      y32 = np.asarray(y, np.float32)
      want = (np.asarray(g, np.float32) * (1.0 - y32) * y32).astype(dt)
      np.testing.assert_array_equal(np.asarray(dx), np.asarray(want))
      _, vjp = jax.vjp(lambda v: am.silu(v, True), x)
      (dx,) = vjp(g)
      x32 = np.asarray(x, np.float32)
      s = np.asarray(am._sigmoid_f32(jnp.asarray(x32)), np.float32)  # pylint: disable=protected-access
      want = (np.asarray(g, np.float32) * s * (1.0 + x32 * (1.0 - s))).astype(dt)
      np.testing.assert_array_equal(np.asarray(dx), np.asarray(want))
      self.assertEqual(dx.dtype, dt)

  def test_primitives_carry_the_accuracy(self):
    x = jnp.asarray(self.x)
    hi = jax.lax.AccuracyMode.HIGHEST
    for fn in (am.exp, am.sigmoid, am.silu):
      self.assertTrue(all(a == hi for _, a in _accuracy_params(lambda v, fn=fn: fn(v, True), x)))
      self.assertTrue(all(a is None for _, a in _accuracy_params(lambda v, fn=fn: fn(v, False), x)))
    self.assertTrue(all(a == hi for _, a in _accuracy_params(lambda v: am.log_softmax(v, -1, True), x)))


class CrossEntropyTest(unittest.TestCase):
  """The loss path: max_utils.cross_entropy_with_logits(accurate=) and the MTP integer-label cross entropy."""

  def setUp(self):
    rng = np.random.default_rng(2)
    self.logits = rng.standard_normal((4, 6, 50)).astype(np.float32) * 3
    self.labels = rng.integers(0, 50, (4, 6))
    self.one_hot = np.eye(50, dtype=np.float32)[self.labels]
    l64 = self.logits.astype(np.float64)
    lse = np.log(np.exp(l64 - l64.max(-1, keepdims=True)).sum(-1)) + l64.max(-1)
    self.ref = lse - np.take_along_axis(l64, self.labels[..., None], -1)[..., 0]

  def test_off_is_the_default_custom_vjp(self):
    lg, oh = jnp.asarray(self.logits), jnp.asarray(self.one_hot)
    a, _ = max_utils.cross_entropy_with_logits(lg, oh, 0.0)
    b, _ = max_utils._cross_entropy_with_logits_default(lg, oh, 0.0)  # pylint: disable=protected-access
    np.testing.assert_array_equal(a, b)

  def test_on_matches_float64_and_has_the_same_gradient(self):
    lg, oh = jnp.asarray(self.logits), jnp.asarray(self.one_hot)
    on, _ = max_utils.cross_entropy_with_logits(lg, oh, 0.0, accurate=True)
    self.assertLessEqual(_ulp_max(on, self.ref), 8)  # one exp, sum, log and subtract on 50 classes
    g_on = jax.grad(lambda v: jnp.sum(max_utils.cross_entropy_with_logits(v, oh, 0.0, accurate=True)[0]))(lg)
    g_off = jax.grad(lambda v: jnp.sum(max_utils.cross_entropy_with_logits(v, oh, 0.0)[0]))(lg)
    np.testing.assert_allclose(np.asarray(g_on), np.asarray(g_off), rtol=1e-5, atol=1e-7)
    hi = jax.lax.AccuracyMode.HIGHEST
    acc = _accuracy_params(lambda v: max_utils.cross_entropy_with_logits(v, oh, 0.0, accurate=True)[0], lg)
    self.assertTrue(acc and all(a == hi for _, a in acc))

  def test_mtp_integer_label_cross_entropy(self):
    lg, lb = jnp.asarray(self.logits), jnp.asarray(self.labels)
    off = mtp._cross_entropy_with_integer_labels(lg, lb)  # pylint: disable=protected-access
    on = mtp._cross_entropy_with_integer_labels(lg, lb, accurate=True)  # pylint: disable=protected-access
    self.assertLessEqual(_ulp_max(on, self.ref), 8)
    np.testing.assert_allclose(np.asarray(on), np.asarray(off), rtol=1e-5)
    hi = jax.lax.AccuracyMode.HIGHEST
    f = mtp._cross_entropy_with_integer_labels  # pylint: disable=protected-access
    acc = _accuracy_params(lambda v: f(v, lb, accurate=True), lg)
    self.assertTrue(acc and all(a == hi for _, a in acc))


class SitesTest(unittest.TestCase):
  """The switch reaches the iHC gates (and the activation lookups used by MlpBlock and the MoE experts)."""

  def test_ihc_gates(self):
    hi = jax.lax.AccuracyMode.HIGHEST
    h = mhc_test.TestMHC()
    for flag in (False, True):
      kw = {"hc_variant": "ihc", "transcendental_accuracy_highest": flag}
      h._setup_mhc(4, extra_kwargs=kw)  # pylint: disable=protected-access
      with nn_partitioning.axis_rules(h.config.logical_axis_rules):
        module = mhc.make_hyper_connections(h.config, h.dim, h.mesh, nnx.Rngs(params=0))
        x = jax.random.normal(jax.random.PRNGKey(5), (2, 7, 4, h.dim), jnp.float32)
        prims = _accuracy_params(module.gates, x)
      if flag:  # no logistic left (its accuracy argument is ignored on TPU); the exp it is built from is HIGHEST
        self.assertFalse([a for name, a in prims if name == "logistic"])
        self.assertTrue([a for name, a in prims if name == "exp"])
        self.assertTrue(all(a == hi for _, a in prims), prims)
      else:
        self.assertTrue([a for name, a in prims if name == "logistic"])
        self.assertTrue(all(a is None for _, a in prims), prims)

  def test_activation_lookup(self):
    on, off = SimpleNamespace(transcendental_accuracy_highest=True), SimpleNamespace(
        transcendental_accuracy_highest=False
    )
    x = jnp.linspace(-8, 8, 64, dtype=jnp.float32)
    hi = jax.lax.AccuracyMode.HIGHEST
    f_on = linears._convert_to_activation_function(linears._accurate("silu", on))  # pylint: disable=protected-access
    f_off = linears._convert_to_activation_function(linears._accurate("silu", off))  # pylint: disable=protected-access
    self.assertTrue(all(a == hi for _, a in _accuracy_params(f_on, x)))
    np.testing.assert_array_equal(f_off(x), jax.nn.silu(x))
    self.assertEqual(linears._accurate("relu", on), "relu")  # pylint: disable=protected-access


class PerSiteTest(unittest.TestCase):
  """accurate_<site> overrides transcendental_accuracy_highest for one sigmoid / silu site; None follows it."""

  def test_site_on(self):
    for glob in (False, True):
      for over in (None, False, True):
        cfg = SimpleNamespace(transcendental_accuracy_highest=glob, accurate_silu_shared=over)
        self.assertEqual(am.site_on(cfg, "silu_shared"), glob if over is None else over)
        self.assertEqual(am.site_on(cfg, "silu_dense"), glob)  # unset site follows the global switch
    with self.assertRaises(AssertionError):
      am.site_on(SimpleNamespace(), "softplus_somewhere")

  def test_mlp_block_site(self):
    cfg = SimpleNamespace(transcendental_accuracy_highest=True, accurate_silu_dense=False)
    x = jnp.linspace(-8, 8, 64, dtype=jnp.float32)
    dense = linears._convert_to_activation_function(linears._accurate("silu", cfg, "silu_dense"))  # pylint: disable=protected-access
    shared = linears._convert_to_activation_function(linears._accurate("silu", cfg, "silu_shared"))  # pylint: disable=protected-access
    np.testing.assert_array_equal(dense(x), jax.nn.silu(x))
    self.assertNotIn("logistic", [n for n, _ in _accuracy_params(shared, x)])

  def test_ihc_and_hc_head_sites(self):
    h = mhc_test.TestMHC()
    for over in (False, True):
      kw = {"hc_variant": "ihc", "transcendental_accuracy_highest": not over, "accurate_sigmoid_ihc": over}
      h._setup_mhc(4, extra_kwargs=kw)  # pylint: disable=protected-access
      with nn_partitioning.axis_rules(h.config.logical_axis_rules):
        module = mhc.make_hyper_connections(h.config, h.dim, h.mesh, nnx.Rngs(params=0))
        x = jax.random.normal(jax.random.PRNGKey(5), (2, 7, 4, h.dim), jnp.float32)
        names = [n for n, _ in _accuracy_params(module.gates, x)]
      self.assertEqual("logistic" not in names, over, (over, names))

  def test_sites_are_wired(self):
    """Each layer asks site_on for its own site (the MoE and MLA modules are too large to build here)."""
    import inspect  # pylint: disable=import-outside-toplevel
    from maxtext.layers import attention_mla, moe  # pylint: disable=import-outside-toplevel

    src = {m: inspect.getsource(m) for m in (mhc, moe, attention_mla)}
    self.assertIn('site_on(self.config, "sigmoid_ihc")', src[mhc])
    self.assertIn('site_on(self.config, "sigmoid_hc_head")', src[mhc])
    self.assertIn('site_on(self.config, "sigmoid_router")', src[moe])
    self.assertIn('site_on(self.config, "silu_routed")', src[moe])
    self.assertIn('accurate_site="silu_shared"', src[moe])
    self.assertIn('site_on(self.config, "sigmoid_attn_gate")', src[attention_mla])


if __name__ == "__main__":
  unittest.main()
