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

"""splash_dsinks_fp32: the attention-sink gradient of the tokamax splash kernel in float32.

tokamax's splash backward computes d_sinks as sum(bf16(-exp(sink - lse)) * o * do) over (q, d) with o and do in the
attention dtype (bf16): every term is rounded to bf16 and the sum and the result are bf16. The same function already
has di = sum_d o * do in float32. With the switch on, d_sinks = -sum_q exp(sink - lse) * di, all in float32; dq, dk,
dv are tokamax's unchanged. This matches GPU (Megatron-Core T1, hy4_mcore: autograd through the float32 softmax, the
sink a float32 parameter): p_sink and the sums are float32 there too. One difference remains: GPU forms
sum_j p_j * dp_j with dp = do . v^T rounded to bf16 per position, TPU uses the identity di = sum_d o * do with o and do
in bf16 (summed in float32): the same quantity, rounded differently (tests/unit/splash_dsinks_fp32_test.py prints both
against float64).

set_enabled() swaps the VJP of tokamax's splash custom_vjp (process-wide) and clears JAX's caches so that no earlier
trace keeps the old backward; set_enabled(False) restores tokamax's own backward.
"""

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as _sk

_ORIG_FWD = _sk._splash_attention_fwd  # pylint: disable=protected-access
_ORIG_BWD = _sk._splash_attention_bwd  # pylint: disable=protected-access
_SINKS_INDEX = 6  # position of the sinks cotangent in the backward's output tuple
_enabled = False


def _bwd_fp32_dsinks(save_residuals, mask_value, is_mqa, config, mask_function, fwd_sparsity, dkv_sparsity, res, grads):
  out = _ORIG_BWD(save_residuals, mask_value, is_mqa, config, mask_function, fwd_sparsity, dkv_sparsity, res, grads)
  _, _, _, _, sinks, o, logsumexp, _, _ = res
  if sinks is None:
    return out
  do = grads[0] if save_residuals else grads
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))
  lse = logsumexp.astype(jnp.float32)
  if config.use_base2_exp:
    lse = lse / _sk.LOG2E
  dsinks = -jnp.sum(jnp.exp(sinks.astype(jnp.float32)[..., None] - lse) * di, axis=-1)
  out = list(out)
  out[_SINKS_INDEX] = dsinks.astype(sinks.dtype)
  return tuple(out)


def is_enabled() -> bool:
  return _enabled


def set_enabled(enabled: bool) -> None:
  """Use the float32 sink gradient (True) or tokamax's own backward (False) for splash calls traced from now on."""
  global _enabled
  if bool(enabled) == _enabled:
    return
  _sk._splash_attention_custom.defvjp(_ORIG_FWD, _bwd_fp32_dsinks if enabled else _ORIG_BWD)  # pylint: disable=protected-access
  _enabled = bool(enabled)
  # tokamax jits the splash call with the config as a static argument: a trace cached before the swap would keep the
  # old backward. Switching is rare (once, when the model is built), so drop the caches.
  jax.clear_caches()
