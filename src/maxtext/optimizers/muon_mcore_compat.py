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

"""optax_muon with three optional Megatron-Core behaviours.

Built only when one of the switches is on; with all of them off optimizers.py keeps calling optax's own `muon`.
The composition below is optax.contrib._muon.muon (optax 0.2.8) with these changes, each behind its own flag:

- mcore_nesterov: Nesterov momentum without bias correction, as Megatron-Core / emerging_optimizers 0.3.0
  (orthogonalized_optimizer.py: buf.lerp_(g, 1 - beta); g' = g.lerp(buf, beta)), i.e.
  mu_t = beta * mu_{t-1} + (1 - beta) * g_t and the update direction is beta * mu_t + (1 - beta) * g_t.
  optax instead uses beta * mu_t / (1 - beta^(t+1)) + (1 - beta) * g_t / (1 - beta^t).
- adam_nesterov: whether the AdamW branch (non-Muon parameters) uses Nesterov (optax NAdam). optax ties it to the
  Muon flag; Megatron-Core's scalar optimizer is plain Adam (False).
- mcore_ns_normalize: before Newton-Schulz, x / max(||x||_F, 1e-7) (torch F.normalize, emerging_optimizers
  muon_utils), instead of optax's x / (||x||_F + eps).
- mcore_ns_bf16: the Newton-Schulz iterations on BF16 tensors, as emerging_optimizers 0.3.0 does at
  fp32_matmul_prec="medium" (Megatron-Core's default muon_fp32_matmul_prec): X is cast to BF16 after the
  normalization and A, B, X are each rounded to BF16 every step; the result goes back to FP32. Off: the iterations
  stay in the update dtype (FP32), with the matmul precision of the surrounding config.
- adam_accurate: the (plain) AdamW branch with accurate float32 arithmetic (scale_by_adam_accurate): TPU lowers
  float32 division, sqrt and pow through approximations (Newton-refined reciprocal and sqrt, 1 - b^t by expm1 here),
  which moves the update
  of tiny tensors (iHC hc_base / hc_scale, attention sinks) by more than GPU's rounding does.
"""

import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from optax._src import alias, base, combine, numerics, transform, utils
from optax.contrib import _muon as optax_muon_lib
from optax.transforms import _masking

MCORE_NS_NORM_EPS = 1e-7


def _bf16_ns_step(x, coeffs):
  """One Newton-Schulz step on BF16 tensors, as emerging_optimizers 0.3.0 newton_schulz_step at
  fp32_matmul_prec="medium": A = X X^T, B = b A + c A A, X = a X + B X; each matmul takes BF16 operands with FP32
  accumulation (torch.addmm's epilogue b*A / a*X in FP32) and each of A, B, X is rounded to BF16 once.
  `x` holds BF16 values in an FP32 array; reduce_precision (not an astype round trip, which XLA may drop under
  excess precision) does the rounding."""
  rnd = lambda v: jax.lax.reduce_precision(v, exponent_bits=8, mantissa_bits=7)
  mm = lambda u, w: jnp.matmul(u.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
  a = rnd(mm(x, x.T))
  b = rnd(coeffs[1] * a + coeffs[2] * mm(a, a))
  return rnd(coeffs[0] * x + mm(b, x))


def _orthogonalize(x, ns_coeffs, ns_steps, eps, dim_num, mcore_ns_normalize, mcore_ns_bf16=False):
  """optax.contrib._muon.orthogonalize_via_newton_schulz (frobenius) with an optional Megatron-Core normalization
  and optional BF16 Newton-Schulz iterations (mcore_ns_bf16)."""

  def one(m):
    transposed = m.shape[0] > m.shape[1]
    if transposed:
      m = m.T
    norm = jnp.linalg.norm(m, ord="fro")
    m = m / jnp.maximum(norm, MCORE_NS_NORM_EPS) if mcore_ns_normalize else m / (norm + eps)
    if mcore_ns_bf16:
      dtype = m.dtype
      coeffs = ns_coeffs.astype(jnp.float32)
      z = jax.lax.reduce_precision(m.astype(jnp.float32), exponent_bits=8, mantissa_bits=7)
      z = jax.lax.fori_loop(0, ns_steps, lambda i, v: _bf16_ns_step(v, coeffs), z, unroll=True)
      m = z.astype(dtype)
    else:
      coeffs = ns_coeffs.astype(m.dtype)
      m = jax.lax.fori_loop(0, ns_steps, lambda i, z: optax_muon_lib._base_ns_iterator(i, z, coeffs), m, unroll=True)
    return m.T if transposed else m

  reshape_fn, inverse_fn = optax_muon_lib._compute_muon_reshape(x, dim_num)  # pylint: disable=protected-access
  return inverse_fn(jax.vmap(one)(reshape_fn(x)))


def _recip(x):
  """1 / x refined by one Newton step, r (2 - x r): TPU lowers float32 division through an approximate reciprocal
  (measured on v5p: x / y up to 2 ULP); one step brings it to about 1 ULP."""
  r = 1.0 / x
  return r * (2.0 - x * r)


def _div(x, y):
  """x / y to about 1 ULP: q = x r with the refined reciprocal, then one correction q + r (x - q y)."""
  r = _recip(y)
  q = x * r
  return q + r * (x - q * y)


def _sqrt(x):
  """sqrt refined by one Newton step, s + (x - s s) / (2 s) (TPU v5p float32 sqrt measured up to 3 ULP); 0 stays 0."""
  s = jnp.sqrt(x)
  safe = jnp.where(s > 0, s, 1.0)
  return jnp.where(s > 0, s + (x - s * s) * (0.5 * _recip(safe)), s)


def _one_minus_pow(decay: float, count):
  """1 - decay^count for a Python float decay, accurate to a few ULP: -expm1(count * log(decay)) with log(decay) taken
  in float64 on the host and expm1 by Kahan's trick from exp / log at accuracy=HIGHEST, (u - 1) y / log(u) with
  u = exp(y). The plain 1 - decay**count loses up to about a hundred ULP on TPU for small counts (pow through an
  approximate exp, then the cancellation in 1 - b^t)."""
  hi = jax.lax.AccuracyMode.HIGHEST
  y = count.astype(jnp.float32) * jnp.float32(math.log(decay))
  u = jax.lax.exp(y, accuracy=hi)
  lu = jax.lax.log(u, accuracy=hi)
  near = (u > 0.5) & (u != 1.0)  # below 0.5 there is no cancellation and u - 1 is already accurate
  kahan = (u - 1.0) * y * _recip(jnp.where(near, lu, 1.0))
  em1 = jnp.where(u == 1.0, y, jnp.where(near, kahan, u - 1.0))
  return -em1


def scale_by_adam_accurate(b1, b2, eps, eps_root, mu_dtype) -> base.GradientTransformation:
  """optax.scale_by_adam (nesterov=False) with the float32 arithmetic made accurate (adam_accurate):
  the bias corrections 1 - b^t through _one_minus_pow, the divisions through _div, the square root through _sqrt.
  Same state (ScaleByAdamState) and the same formula, m_hat / (sqrt(v_hat + eps_root) + eps)."""
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    return transform.ScaleByAdamState(
        count=jnp.zeros([], jnp.int32),
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        nu=optax.tree.zeros_like(params),
    )

  def update_fn(updates, state, params=None):
    del params
    mu = optax.tree.update_moment(updates, state.mu, b1, 1)
    nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    c1, c2 = _one_minus_pow(b1, count_inc), _one_minus_pow(b2, count_inc)
    # Same order as the fused Adam kernels: m_hat = m / c1, v_hat = v / c2, m_hat / (sqrt(v_hat + eps_root) + eps)
    new = jax.tree.map(
        lambda m, v: None
        if m is None
        else _div(_div(m, c1.astype(m.dtype)), _sqrt(_div(v, c2.astype(v.dtype)) + eps_root) + eps),
        mu,
        nu,
        is_leaf=lambda x: x is None,
    )
    return new, transform.ScaleByAdamState(
        count=count_inc, mu=optax.tree.cast(mu, mu_dtype), nu=optax.tree.cast_like(nu, state.nu)
    )

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_muon_compat(
    ns_coeffs,
    ns_steps: int,
    beta: float,
    eps: float,
    mu_dtype,
    nesterov: bool,
    mcore_nesterov: bool,
    mcore_ns_normalize: bool,
    weight_dimension_numbers,
    mcore_ns_bf16: bool = False,
) -> base.GradientTransformation:
  """optax.contrib._muon.scale_by_muon (frobenius preconditioning, not adaptive) with the two Muon switches."""
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    return optax_muon_lib.MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=optax.tree.zeros_like(params, dtype=mu_dtype),
        ns_coeffs=jnp.asarray(ns_coeffs),
    )

  def update_fn(updates, state, params=None):
    del params
    dim_nums = weight_dimension_numbers(updates) if callable(weight_dimension_numbers) else weight_dimension_numbers
    mu = optax.tree.update_moment(updates, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov and mcore_nesterov:
      mu_hat = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, mu, updates)
    elif nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(mu, beta, numerics.safe_increment(count_inc)),
          optax.tree.bias_correction(updates, beta, count_inc),
      )
    elif mcore_nesterov:
      mu_hat = mu
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)
    updates = jax.tree.map(
        lambda x, d: _orthogonalize(x, state.ns_coeffs, ns_steps, eps, d, mcore_ns_normalize, mcore_ns_bf16),
        mu_hat,
        dim_nums,
        is_leaf=optax_muon_lib._is_weight_dim_nums,  # pylint: disable=protected-access
    )
    return updates, optax_muon_lib.MuonState(count=count_inc, mu=optax.tree.cast(mu, mu_dtype), ns_coeffs=state.ns_coeffs)

  return base.GradientTransformation(init_fn, update_fn)


def muon_compat(
    learning_rate,
    ns_coeffs,
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Any] = None,
    mu_dtype=None,
    *,
    nesterov: bool = True,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    adam_learning_rate=None,
    muon_weight_dimension_numbers: Any | Callable = None,
    consistent_rms: float | None = None,
    mcore_nesterov: bool = False,
    adam_nesterov: bool | None = None,
    mcore_ns_normalize: bool = False,
    mcore_ns_bf16: bool = False,
    adam_accurate: bool = False,
) -> base.GradientTransformation:
  """Same partition and chain as optax.contrib._muon.muon; see the module docstring for the switches."""
  if adam_learning_rate is None:
    adam_learning_rate = learning_rate
  if adam_nesterov is None:
    adam_nesterov = nesterov
  if adam_accurate and adam_nesterov:
    raise ValueError("adam_accurate implements plain Adam only (set muon_adam_plain).")
  if muon_weight_dimension_numbers is None:
    raise ValueError("muon_compat needs muon_weight_dimension_numbers")
  is_dn = optax_muon_lib._is_weight_dim_nums  # pylint: disable=protected-access

  def param_labels(params):
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    return jax.tree.map(
        lambda d, x: jax.tree.map(lambda _: "muon" if d is not None else "adam", x),
        dim_nums,
        params,
        is_leaf=lambda x: x is None or is_dn(x),
    )

  def muon_dim_nums_fn(params):
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
    is_leaf = lambda x: x is None or is_dn(x) or isinstance(x, _masking.MaskedNode)
    return jax.tree.map(
        lambda d, sub: jax.tree.map(lambda m: d if m else _masking.MaskedNode(), sub), dim_nums, mask, is_leaf=is_leaf
    )

  return combine.partition(
      transforms={
          "muon": combine.chain(
              scale_by_muon_compat(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  mcore_nesterov=mcore_nesterov,
                  mcore_ns_normalize=mcore_ns_normalize,
                  weight_dimension_numbers=muon_dim_nums_fn,
                  mcore_ns_bf16=mcore_ns_bf16,
              ),
              optax_muon_lib.scale_by_shape(weight_dimension_numbers=muon_dim_nums_fn, consistent_rms=consistent_rms),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          "adam": (
              combine.chain(
                  scale_by_adam_accurate(adam_b1, adam_b2, eps, adam_eps_root, mu_dtype),
                  transform.add_decayed_weights(adam_weight_decay),
                  transform.scale_by_learning_rate(adam_learning_rate),
              )
              if adam_accurate
              else alias.adamw(
                  learning_rate=adam_learning_rate,
                  b1=adam_b1,
                  b2=adam_b2,
                  eps=eps,
                  eps_root=adam_eps_root,
                  weight_decay=adam_weight_decay,
                  mu_dtype=mu_dtype,
                  nesterov=adam_nesterov,
              )
          ),
      },
      param_labels=param_labels,
  )
