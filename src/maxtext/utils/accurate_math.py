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

"""FP32 transcendentals with `accuracy=HIGHEST` (transcendental_accuracy_highest).

On TPU, XLA's default float32 exp / log / logistic are fast approximations: the JAX "Numerical accuracy" page lists
TPU v5p maxima of 109.5 / 62 / 124 ULP (64.5 / 2.5 / 65.5 on v6e), against 1-4 ULP for CUDA. `lax.exp`, `lax.log`
take `accuracy=lax.AccuracyMode.HIGHEST`, which brings exp / log on TPU v5p to 1 / 3 ULP (measured). `lax.logistic`
takes the same argument but on TPU v5p (JAX 0.11.1) its result is bit-identical to the default, still 83 ULP, so
sigmoid / silu are built from the HIGHEST exp and a Newton-refined reciprocal instead. Each helper below takes
`accurate`: False calls the usual jax function unchanged; True uses the accurate version.

softmax / log_softmax follow jax.nn.softmax / jax.nn.log_softmax (max subtracted with stop_gradient, exp, sum) with
the accurate exp / log; they are differentiated through those ops instead of jax.nn's custom JVP.
"""

import jax
import jax.numpy as jnp
from jax import lax

_HI = lax.AccuracyMode.HIGHEST


def exp(x, accurate: bool):
  return lax.exp(x, accuracy=_HI) if accurate else jnp.exp(x)


def log(x, accurate):
  """accurate: False = jnp.log; True = lax.log at accuracy=HIGHEST (TPU v5p: 3 ULP); "freebsd" = log_freebsd (1 ULP)."""
  if accurate == "freebsd":
    return log_freebsd(x)
  return lax.log(x, accuracy=_HI) if accurate else jnp.log(x)


def log_mode(config, accurate):
  """The `accurate` value for the float32 log sites: "freebsd" when transcendental_log_freebsd is on and the site is on
  its accurate path (transcendental_accuracy_highest), otherwise `accurate` unchanged."""
  if accurate and getattr(config, "transcendental_log_freebsd", False):
    return "freebsd"
  return accurate


# FreeBSD msun e_logf.c (__ieee754_logf), Sun Microsystems 1993 / Ian Lance Taylor's float version: constants as there.
_LN2_HI, _LN2_LO = 6.9313812256e-01, 9.0580006145e-06
_LG = (6.6666668653e-01, 4.0000000596e-01, 2.8571429849e-01, 2.2222198546e-01, 1.8183572590e-01, 1.5313838422e-01,
       1.4798198640e-01)


def _div_exact(a, b):
  """a / b to about 0.5 ULP in float32: refined reciprocal, then one correction q + r (a - q b). TPU's float32 division
  goes through an approximate reciprocal (v5p: up to 2 ULP), which would show up in s = f / (2 + f) below."""
  r = 1.0 / b
  r = r * (2.0 - b * r)
  q = a * r
  return q + r * (a - q * b)


def _log_freebsd_f32(x):
  """e_logf.c in float32 arithmetic: x = 2^k (1 + f), f in [sqrt(2)/2 - 1, sqrt(2) - 1); s = f / (2 + f);
  log(1 + f) = f - s (f - R(s^2)) with a degree-7 R; k ln2 split into a high part exact in float32 and a low part.
  The source states an error below 1 ULP; measured on TPU v5p: at most 1 ULP over every float32 in [0.5, 4) and
  over log-uniform samples in [1e-37, 1e37] and [1, 4096). Zero, subnormals, negatives, inf and nan go to jnp.log."""
  f32 = jnp.float32
  ix = lax.bitcast_convert_type(x, jnp.int32)
  k = (ix >> 23) - 127  # subnormal, zero and negative inputs are masked out at the end

  ix = ix & 0x007FFFFF
  i = (ix + (0x95F64 << 3)) & 0x800000
  xn = lax.bitcast_convert_type(ix | (i ^ 0x3F800000), f32)  # normalize x or x / 2
  k = k + (i >> 23)
  f = xn - f32(1.0)
  dk = k.astype(f32)
  ln2_hi, ln2_lo = f32(_LN2_HI), f32(_LN2_LO)
  # |f| < 2^-20
  rt = f * f * (f32(0.5) - f32(0.33333333333333333) * f)
  tiny_v = jnp.where(k == 0, f - rt, dk * ln2_hi - ((rt - dk * ln2_lo) - f))
  tiny_v = jnp.where(f == 0, jnp.where(k == 0, f32(0.0), dk * ln2_hi + dk * ln2_lo), tiny_v)
  s = _div_exact(f, f32(2.0) + f)
  z = s * s
  w = z * z
  lg = [f32(c) for c in _LG]
  t1 = w * (lg[1] + w * (lg[3] + w * lg[5]))
  t2 = z * (lg[0] + w * (lg[2] + w * (lg[4] + w * lg[6])))
  big = ((ix - (0x6147A << 3)) | ((0x6B851 << 3) - ix)) > 0
  r = t2 + t1
  hfsq = f32(0.5) * f * f
  a1 = jnp.where(k == 0, f - (hfsq - s * (hfsq + r)), dk * ln2_hi - ((hfsq - (s * (hfsq + r) + dk * ln2_lo)) - f))
  a2 = jnp.where(k == 0, f - s * (f - r), dk * ln2_hi - ((s * (f - r) - dk * ln2_lo) - f))
  v = jnp.where((0x007FFFFF & (15 + ix)) < 16, tiny_v, jnp.where(big, a1, a2))
  # Subnormal inputs go to jnp.log as well: TPU flushes them to zero, and e_logf's rescaling by 2^25 would see the
  # flushed product; the sites using this (probabilities clamped to >= 1e-10, sums >= 1) never see them.
  return jnp.where((x >= f32(1.1754943508222875e-38)) & jnp.isfinite(x), v, jnp.log(x))


@jax.custom_jvp
def log_freebsd(x):
  """Natural log through _log_freebsd_f32 in float32, result in x.dtype (float32 inputs: about 1 ULP, as CUDA logf).
  The derivative is lax.log's own rule, t / x (the bit manipulation above has no derivative). Mathematically the
  gradients are those of jnp.log / lax.log (bit for bit in the CPU unit tests); on TPU the surrounding program can
  fuse differently, which moved some indexer gradients by ~1e-4 relative on hy4-mini."""
  return _log_freebsd_f32(x.astype(jnp.float32)).astype(x.dtype)


@log_freebsd.defjvp
def _log_freebsd_jvp(primals, tangents):
  (x,), (t,) = primals, tangents
  return log_freebsd(x), t / x


def expm1(x, accurate: bool):
  return lax.expm1(x, accuracy=_HI) if accurate else jnp.expm1(x)


def _recip(x):
  """1 / x refined by one Newton step, r (2 - x r): TPU's float32 division goes through an approximate reciprocal."""
  r = 1.0 / x
  return r * (2.0 - x * r)


def _sigmoid_f32(x):
  """sigmoid in float32 from exp(-|x|) at accuracy=HIGHEST (never overflows): 1 / (1 + e) for x >= 0, e / (1 + e)
  below. About 2-3 ULP: exp 1 ULP, the add and the product half an ULP each, the refined reciprocal about 1."""
  e = lax.exp(-jnp.abs(x), accuracy=_HI)
  r = _recip(1.0 + e)
  return jnp.where(x >= 0, r, e * r)


@jax.custom_vjp
def _accurate_sigmoid(x):
  return _sigmoid_f32(x.astype(jnp.float32)).astype(x.dtype)


def _accurate_sigmoid_fwd(x):
  y = _accurate_sigmoid(x)
  return y, y


def _accurate_sigmoid_bwd(y, g):
  # PyTorch's sigmoid_backward (CUDA): grad * (1 - y) * y in float32 opmath, from the stored output y (rounded to the
  # output dtype), result in the input dtype. Autodiff of the composition above would give a different expression.
  y32 = y.astype(jnp.float32)
  return ((g.astype(jnp.float32) * (1.0 - y32) * y32).astype(y.dtype),)


_accurate_sigmoid.defvjp(_accurate_sigmoid_fwd, _accurate_sigmoid_bwd)


@jax.custom_vjp
def _accurate_silu(x):
  x32 = x.astype(jnp.float32)
  return (x32 * _sigmoid_f32(x32)).astype(x.dtype)


def _accurate_silu_fwd(x):
  return _accurate_silu(x), x


def _accurate_silu_bwd(x, g):
  # PyTorch's silu_backward (CUDA): s = 1 / (1 + exp(-x)) recomputed from the input in float32, then
  # grad * s * (1 + x * (1 - s)); result in the input dtype.
  x32 = x.astype(jnp.float32)
  s = _sigmoid_f32(x32)
  return ((g.astype(jnp.float32) * s * (1.0 + x32 * (1.0 - s))).astype(x.dtype),)


_accurate_silu.defvjp(_accurate_silu_fwd, _accurate_silu_bwd)


def sigmoid(x, accurate: bool):
  """accurate: float32 forward from the HIGHEST exp, rounded once to x.dtype; backward in PyTorch's analytic form."""
  return _accurate_sigmoid(x) if accurate else jax.nn.sigmoid(x)


def silu(x, accurate: bool):
  """accurate: float32 forward x * sigmoid(x), rounded once to x.dtype; backward in PyTorch's analytic form."""
  return _accurate_silu(x) if accurate else jax.nn.silu(x)


def softmax(x, axis: int = -1, accurate: bool = False):
  if not accurate:
    return jax.nn.softmax(x, axis=axis)
  x_max = jnp.max(x, axis, keepdims=True)
  unnormalized = exp(x - lax.stop_gradient(x_max), True)
  return unnormalized / jnp.sum(unnormalized, axis, keepdims=True)


def log_softmax(x, axis: int = -1, accurate=False):
  """accurate: False = jax.nn.log_softmax; True = accurate exp / log; "freebsd" = accurate exp, log_freebsd."""
  if not accurate:
    return jax.nn.log_softmax(x, axis=axis)
  shifted = x - lax.stop_gradient(jnp.max(x, axis, keepdims=True))
  return shifted - log(jnp.sum(exp(shifted, True), axis, keepdims=True), accurate)


SITES = (
    "sigmoid_ihc",  # iHC pre / post gates (mhc.py)
    "sigmoid_hc_head",  # hc_head gate (mhc.py)
    "sigmoid_router",  # routed score function (moe.py GateLogit)
    "sigmoid_attn_gate",  # MLA elementwise output gate (attention_mla.py)
    "silu_dense",  # dense-layer MLP (linears.MlpBlock)
    "silu_shared",  # shared expert (linears.MlpBlock inside the MoE block)
    "silu_routed",  # routed experts (moe.py)
)


def site_on(config, site: str) -> bool:
  """Whether the accurate sigmoid / silu is used at `site`: config.accurate_<site> when set (True / False), else
  transcendental_accuracy_highest (the global switch, which also covers the exp / log sites)."""
  assert site in SITES, site
  v = getattr(config, f"accurate_{site}", None)
  return bool(getattr(config, "transcendental_accuracy_highest", False)) if v is None else bool(v)


def activation(name: str, accurate: bool):
  """jax.nn activation by name, with the accurate sigmoid / silu when `accurate`; other names are unchanged."""
  if accurate and name == "silu":
    return lambda v: silu(v, True)
  if accurate and name == "sigmoid":
    return lambda v: sigmoid(v, True)
  return None
