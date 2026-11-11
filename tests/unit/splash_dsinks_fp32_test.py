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

"""splash_dsinks_fp32: the tokamax splash sink gradient in float32, against a float64 reference (Pallas interpret mode
on CPU). Off is tokamax's own backward (bf16 terms, bf16 sum); dq / dk / dv are the same bits off and on."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as mask_lib

from maxtext.kernels import splash_dsinks_fp32

H, S, D = 2, 512, 128


def _reference_dsinks(q, k, v, sinks, do, o_given=None, lse_given=None):
  """float64: causal softmax with a per-head sink logit; d sink = -sum_q p_sink * (o . do)."""
  q, k, v, do = (np.asarray(x, np.float64) for x in (q, k, v, do))
  s = np.einsum("hqd,hkd->hqk", q, k)
  s = np.where(np.tril(np.ones((S, S), bool))[None], s, -np.inf)
  m = np.maximum(s.max(-1), np.asarray(sinks, np.float64)[:, None])
  e = np.exp(s - m[..., None])
  es = np.exp(np.asarray(sinks, np.float64)[:, None] - m)
  den = e.sum(-1) + es
  o = np.einsum("hqk,hkd->hqd", e / den[..., None], v) if o_given is None else np.asarray(o_given, np.float64)
  di = np.sum(o * do, -1)
  p_sink = es / den if lse_given is None else np.exp(np.asarray(sinks, np.float64)[:, None] - lse_given)
  return -np.sum(p_sink * di, -1)


def _reference_dsinks_gpu_form(q, k, v, sinks, do):
  """float64 except dp = do . v^T rounded to bf16 once per position, as Megatron-Core's autograd through the FP32
  softmax computes it: d sink = -sum_q p_sink * sum_j p_j dp_j."""
  q, k, v, do = (np.asarray(x, np.float64) for x in (q, k, v, do))
  s = np.einsum("hqd,hkd->hqk", q, k)
  s = np.where(np.tril(np.ones((S, S), bool))[None], s, -np.inf)
  m = np.maximum(s.max(-1), np.asarray(sinks, np.float64)[:, None])
  e = np.exp(s - m[..., None])
  es = np.exp(np.asarray(sinks, np.float64)[:, None] - m)
  den = e.sum(-1) + es
  dp = np.asarray(np.einsum("hqd,hkd->hqk", do, v), np.float32).astype(jnp.bfloat16).astype(np.float64)
  return -np.sum(es / den * np.sum(e / den[..., None] * dp, -1), -1)


class SplashDsinksFp32Test(unittest.TestCase):

  def _grads(self, enabled, q, k, v, sinks, do):
    splash_dsinks_fp32.set_enabled(enabled)
    try:
      cfg = splash.SplashConfig(
          block_q=128,
          block_kv=128,
          block_kv_compute=128,
          block_q_dkv=128,
          block_kv_dkv=128,
          block_kv_dkv_compute=128,
          interpret=True,
      )
      mask = mask_lib.CausalMask((S, S))
      attn = splash.make_splash_mha_single_device(mask, config=cfg)
      attn_res = splash.make_splash_mha_single_device(mask, config=cfg, save_residuals=True)
      o, stats = attn_res(q, k, v, None, sinks)
      _, vjp = jax.vjp(lambda q_, k_, v_, s_: attn(q_, k_, v_, None, s_), q, k, v, sinks)
      lse = np.asarray(stats["logsumexp"], np.float64)  # natural log (tokamax converts the base-2 value for the stats)
      return vjp(do) + (o, lse)
    finally:
      splash_dsinks_fp32.set_enabled(False)

  def test_fp32_sink_gradient(self):
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((H, S, D)) * 0.3, jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((H, S, D)) * 0.3, jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((H, S, D)), jnp.bfloat16)
    do = jnp.asarray(rng.standard_normal((H, S, D)), jnp.bfloat16)
    sinks = jnp.asarray(rng.standard_normal(H), jnp.float32)
    want = _reference_dsinks(q, k, v, sinks, do)
    off = self._grads(False, q, k, v, sinks, do)
    on = self._grads(True, q, k, v, sinks, do)
    # the kernel's own bf16 o and its logsumexp (its q.k is not float64-exact; lse differs by ~1e-3)
    want_o16 = _reference_dsinks(q, k, v, sinks, do, o_given=on[4], lse_given=on[5])
    gpu_form = _reference_dsinks_gpu_form(q, k, v, sinks, do)

    def rel(x, ref):
      return float(np.linalg.norm(np.asarray(x, np.float64) - ref) / np.linalg.norm(ref))

    self.assertEqual(on[3].dtype, jnp.float32)
    # given the kernel's bf16 o, on is float32-exact; off rounds every term and the sum to bf16
    self.assertLess(rel(on[3], want_o16), 1e-5)
    self.assertGreater(rel(off[3], want_o16), 30 * rel(on[3], want_o16))
    self.assertLess(rel(on[3], want), rel(off[3], want))
    # the remaining difference to the GPU's form (dp rounded to bf16 per position vs o, do in bf16), both vs float64
    print(
        f"[dsinks] vs float64: tpu on {rel(on[3], want):.2e}, tpu off {rel(off[3], want):.2e}, "
        f"gpu form {rel(gpu_form, want):.2e}; given the kernel's o / lse: on {rel(on[3], want_o16):.2e}, "
        f"off {rel(off[3], want_o16):.2e}"
    )
    for a, b in zip(on[:3], off[:3]):  # dq, dk, dv untouched
      np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


if __name__ == "__main__":
  unittest.main()
