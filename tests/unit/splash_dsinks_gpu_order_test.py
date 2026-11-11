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

"""splash_dsinks_gpu_order: the sink gradient in Megatron-Core's order, around the tokamax splash kernel (Pallas
interpret mode on CPU).

- gpu_order_dsinks against the GPU form written out in numpy float32 (dp rounded to bf16 per position): near bit-equal;
- with_gpu_order_sink_grad around splash (kernel fed stop_gradient(sinks)): the output and dq / dk / dv are the kernel's
  own bits, d_sink is gpu_order_dsinks; against float64 its error equals the GPU form's (and is below
  splash_dsinks_fp32's and tokamax's);
- a sparse (non-causal-block) mask is honored.
"""

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

from maxtext.kernels import splash_dsinks_fp32
from maxtext.kernels.splash_dsinks_gpu_order import gpu_order_dsinks, with_gpu_order_sink_grad

H, S, D = 2, 512, 128


def _gpu_form(q, k, v, sinks, do, allowed, dtype):
  """The GPU order in numpy: scores, softmax over [allowed keys, sink], dp = bf16(do . v^T), sums, all in `dtype`."""
  q, k, v, do = (np.asarray(x, np.float32).astype(dtype) for x in (q, k, v, do))
  sk = np.asarray(sinks, np.float32).astype(dtype)
  s = np.einsum("hqd,hkd->hqk", q, k)
  s = np.where(allowed[None], s, -np.inf)
  m = np.maximum(s.max(-1), sk[:, None])
  den = np.exp(s - m[..., None]).sum(-1) + np.exp(sk[:, None] - m)
  lse = m + np.log(den)
  p = np.exp(s - lse[..., None])
  ps = np.exp(sk[:, None] - lse)
  dp = np.einsum("hqd,hkd->hqk", np.asarray(do, np.float32), np.asarray(v, np.float32))
  dp = dp.astype(ml_dtypes.bfloat16).astype(dtype)
  return -np.sum(ps * np.sum(p * dp, -1), -1)


def _exact(q, k, v, sinks, do, allowed):
  """float64 throughout (no bf16 rounding of dp): the true d_sink for these bf16 inputs."""
  q, k, v, do = (np.asarray(x, np.float64) for x in (q, k, v, do))
  sk = np.asarray(sinks, np.float64)
  s = np.where(allowed[None], np.einsum("hqd,hkd->hqk", q, k), -np.inf)
  m = np.maximum(s.max(-1), sk[:, None])
  den = np.exp(s - m[..., None]).sum(-1) + np.exp(sk[:, None] - m)
  p = np.exp(s - m[..., None]) / den[..., None]
  ps = np.exp(sk[:, None] - m) / den
  return -np.sum(ps * np.sum(p * np.einsum("hqd,hkd->hqk", do, v), -1), -1)


def _rel(x, ref):
  return float(np.linalg.norm(np.asarray(x, np.float64) - ref) / np.linalg.norm(ref))


class SplashDsinksGpuOrderTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    rng = np.random.default_rng(0)
    cls.q = jnp.asarray(rng.standard_normal((H, S, D)) * 0.3, jnp.bfloat16)
    cls.k = jnp.asarray(rng.standard_normal((H, S, D)) * 0.3, jnp.bfloat16)
    cls.v = jnp.asarray(rng.standard_normal((H, S, D)), jnp.bfloat16)
    cls.do = jnp.asarray(rng.standard_normal((H, S, D)), jnp.bfloat16)
    cls.sinks = jnp.asarray(rng.standard_normal(H), jnp.float32)
    cls.causal = np.tril(np.ones((S, S), bool))
    cls.cfg = splash.SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        interpret=True,
    )

  def test_matches_gpu_form_numpy(self):
    for allowed in (self.causal, self.causal & (np.random.default_rng(3).random((S, S)) < 0.3) | np.eye(S, dtype=bool)):
      got = np.asarray(
          jax.jit(gpu_order_dsinks, static_argnums=6)(
              self.sinks, self.q, self.k, self.v, jnp.asarray(allowed), self.do, 128
          )
      )
      want32 = _gpu_form(self.q, self.k, self.v, self.sinks, self.do, allowed, np.float32)
      want64 = _gpu_form(self.q, self.k, self.v, self.sinks, self.do, allowed, np.float64)
      print(f"[gpu order] vs numpy float32 GPU form {_rel(got, want32):.2e}, vs float64 GPU form {_rel(got, want64):.2e}")
      self.assertLess(_rel(got, want64), 1e-5)
      self.assertLess(_rel(got, want32), 1e-5)

  def _kernel_grads(self, wrap, fp32=False):
    splash_dsinks_fp32.set_enabled(fp32)
    try:
      attn = splash.make_splash_mha_single_device(mask_lib.CausalMask((S, S)), config=self.cfg)
      allowed = jnp.asarray(self.causal)

      def f(q_, k_, v_, s_):
        if not wrap:
          return attn(q_, k_, v_, None, s_)
        o = attn(q_, k_, v_, None, jax.lax.stop_gradient(s_))
        return with_gpu_order_sink_grad(o, s_, q_, k_, v_, allowed, 128)

      o, vjp = jax.vjp(f, self.q, self.k, self.v, self.sinks)
      return (o,) + vjp(self.do)
    finally:
      splash_dsinks_fp32.set_enabled(False)

  def test_around_splash(self):
    plain = self._kernel_grads(False)
    fp32 = self._kernel_grads(False, fp32=True)
    gpu = self._kernel_grads(True)
    for a, b in zip(gpu[:4], plain[:4]):  # output, dq, dk, dv: the kernel's own bits
      np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
    direct = gpu_order_dsinks(self.sinks, self.q, self.k, self.v, jnp.asarray(self.causal), self.do, 128)
    np.testing.assert_array_equal(np.asarray(gpu[4]), np.asarray(direct))
    exact = _exact(self.q, self.k, self.v, self.sinks, self.do, self.causal)
    gpu_form = _gpu_form(self.q, self.k, self.v, self.sinks, self.do, self.causal, np.float64)
    r = {n: _rel(x[4], exact) for n, x in (("tokamax", plain), ("fp32 (M42)", fp32), ("gpu order (M42b)", gpu))}
    print(f"[dsinks vs float64] {r}, GPU form {_rel(gpu_form, exact):.2e}")
    self.assertLess(abs(r["gpu order (M42b)"] - _rel(gpu_form, exact)), 0.02 * _rel(gpu_form, exact))
    self.assertLess(r["gpu order (M42b)"], r["fp32 (M42)"])
    self.assertLess(r["gpu order (M42b)"], r["tokamax"])


if __name__ == "__main__":
  unittest.main()
