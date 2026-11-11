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
"""Fused Pallas TPU kernels for identity hyper-connections (iHC).

Computes the same function as the XLA path of `IdentityHyperConnections` (layers/mhc.py), per token with the k
streams flattened into `flat` of size k * d and n = 2k gate columns:

  raw   = flat @ alpha                                  alpha = [pre_alpha, post_alpha], (k*d, n) float32
  inv   = rsqrt(mean(flat**2) + rms_epsilon)
  gates = mult * sigmoid(raw * inv * scale + beta) + hc_eps    mult = 1 for pre, post_magnitude for post
  layer_input = sum_s pre_s * x_s                       (float32, cast to the activation dtype)
  out_s       = x_s + post_s * layer_output             (float32, cast to the activation dtype)

Everything is float32 except the gate projection, which runs on the MXU as bfloat16 products accumulated in float32
without losing float32 precision: `flat` is a bfloat16 activation cast up (exact in bfloat16), and every float32
operand on the other side is split exactly into three bfloat16 pieces (hi + mid + lo carries all 24 mantissa bits),
so each product is exact and only the float32 accumulation order differs from a float32 matmul. The pieces are
stacked along a dimension the MXU pads anyway (output columns forward, contraction rows backward), so the split
costs one MXU pass instead of three or nine.

`ihc_pre` returns x unchanged next to the branch input, and `ihc_post` takes that copy: the gradient that reaches x
through `ihc_post` then arrives as a cotangent of `ihc_pre`, whose backward kernel adds it while it writes d_x, so the
residual-stream gradient is read and written once (the same trick as the mHC-lite kernels in kernels/mhc).

Kernels see tokens flattened to rows: x is (tokens, k * d), a free reshape of (batch, seq, k, d).
"""

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

_F32 = jnp.float32
_BF16 = jnp.bfloat16
# Rows of the stacked bfloat16 RHS: forward 3 pieces x n gate columns, backward 3 x 3 pieces x n.
_FWD_RHS_ROWS = 32
_BWD_RHS_ROWS = 128
_COEF_ROWS = 8  # rows: scale, beta, mult, then zero padding up to one float32 sublane tile


@dataclasses.dataclass(frozen=True)
class IhcKernelConfig:
  """Static configuration of the iHC kernels.

  mesh / token_axes: when mesh is set, every kernel runs under a shard_map that splits the batch and sequence axes
  over token_axes (two PartitionSpec entries) and keeps the stream and embedding axes whole.
  """

  rms_epsilon: float
  hc_eps: float
  post_magnitude: float
  block_size: int = 128
  bwd_block_size: int = 64
  vmem_limit_bytes: int = 56 * 1024 * 1024
  mesh: Mesh | None = None
  token_axes: tuple = (None, None)
  interpret: bool = False


def _top_bf16_bits(v):
  """v with the low 16 bits of its float32 pattern cleared: v truncated to bfloat16 precision, still float32."""
  bits = jax.lax.bitcast_convert_type(v, jnp.int32) & jnp.int32(-65536)  # 0xFFFF0000
  return jax.lax.bitcast_convert_type(bits, _F32)


def _split3(a):
  """Splits float32 `a` into bfloat16 hi, mid, lo with hi + mid + lo == a (exact unless it underflows bfloat16).

  The pieces come from truncating the bit pattern (hi = top 8 significand bits, mid = top 8 of the remainder, lo = the
  remaining <= 8 bits), so each is exact in bfloat16 and the final casts round nothing. Rounding with a
  float32 -> bfloat16 -> float32 round trip instead does not survive XLA on TPU: with excess precision allowed (the
  default) the round trip is folded away, mid and lo come out zero, and the projection drops to one bfloat16 pass.
  """
  a = a.astype(_F32)
  hi = _top_bf16_bits(a)
  r = a - hi
  mid = _top_bf16_bits(r)
  lo = r - mid
  return hi.astype(_BF16), mid.astype(_BF16), lo.astype(_BF16)


def _gate_rhs(alpha):
  """Stacked bfloat16 pieces of alpha^T: forward (32, k*d) rows [hi; mid; lo], backward (128, k*d) that block x 3."""
  n, kd = alpha.shape[1], alpha.shape[0]
  pieces = [p.T for p in _split3(alpha.astype(_F32))]  # 3 x (n, k*d)
  fwd = jnp.concatenate(pieces + [jnp.zeros((_FWD_RHS_ROWS - 3 * n, kd), _BF16)], axis=0)
  bwd = jnp.concatenate(pieces * 3 + [jnp.zeros((_BWD_RHS_ROWS - 9 * n, kd), _BF16)], axis=0)
  return fwd, bwd


def _placements(n):
  """Constant 0/1 matrices that lay out the three pieces of a (rows, n) value as kernel matmul operands.

  bwd[p]: (n, 128), piece p into columns [3p*n, (3p+3)*n) three times, matching the backward RHS rows
  [hi; mid; lo] x 3, so one contraction sums all nine piece products.
  acc[p]: (n, 32), piece p into columns [p*n, (p+1)*n), for the d_alpha accumulator.
  """
  eye = jnp.eye(n, dtype=_F32)
  bwd, acc = [], []
  for p in range(3):
    m = jnp.zeros((n, _BWD_RHS_ROWS), _F32)
    for col in range(3 * p, 3 * p + 3):
      m = m.at[:, col * n : (col + 1) * n].set(eye)
    bwd.append(m)
    acc.append(jnp.zeros((n, _FWD_RHS_ROWS), _F32).at[:, p * n : (p + 1) * n].set(eye))
  return bwd, acc


def _coef(scale, beta, k, post_magnitude):
  """(8, n) float32: row 0 gate scale, row 1 gate bias, row 2 gate multiplier (1 pre, post_magnitude post)."""
  mult = jnp.concatenate([jnp.ones((k,), _F32), jnp.full((k,), post_magnitude, _F32)])
  rows = jnp.stack([scale.astype(_F32), beta.astype(_F32), mult])
  return jnp.concatenate([rows, jnp.zeros((_COEF_ROWS - 3, 2 * k), _F32)], axis=0)


def _whole(shape):
  return pl.BlockSpec(shape, lambda i: (0,) * len(shape))


def _rows(block, width):
  return pl.BlockSpec((block, width), lambda i: (i, 0))


def _compiler_params(cfg, semantics):
  return pltpu.CompilerParams(dimension_semantics=(semantics,), vmem_limit_bytes=cfg.vmem_limit_bytes)


# Token rows per block: a multiple of the bfloat16 sublane tile (16), since x / d_x blocks are bfloat16.
_ROW_ALIGN = 16
MAX_STREAMS = _FWD_RHS_ROWS // 6  # forward RHS holds 3 pieces x 2k gate columns in 32 rows


def tokens_supported(tokens: int, cfg: "IhcKernelConfig") -> bool:
  """Whether `tokens` rows per device tile into both kernel block sizes."""
  return all(b >= _ROW_ALIGN and b % _ROW_ALIGN == 0 and tokens % b == 0 for b in (cfg.block_size, cfg.bwd_block_size))


def _check_tokens(tokens, block, name):
  if block < _ROW_ALIGN or block % _ROW_ALIGN or tokens % block:
    raise ValueError(
        f"iHC kernel: per-device token count {tokens} must be a multiple of {name}={block} (itself a multiple of {_ROW_ALIGN})."
    )


# ---------------------------------------------------------------------------------------------------------------------
# Kernels. x rows are tokens; stream s of a row is lanes [s*d, (s+1)*d).
# ---------------------------------------------------------------------------------------------------------------------


def _pre_fwd_kernel(x_ref, rhs_ref, coef_ref, li_ref, gates_ref, raw_ref, inv_ref, *, k, d, cfg):
  """Gates, then the weighted stream sum. Outputs layer_input, gates (pre | post), raw projection and inv_rms."""
  n = 2 * k
  sumsq, proj = None, None
  for s in range(k):
    xs = x_ref[:, s * d : (s + 1) * d]
    xf = xs.astype(_F32)
    part = jnp.sum(xf * xf, axis=-1, keepdims=True)
    pp = jax.lax.dot_general(
        xs.astype(_BF16), rhs_ref[:, s * d : (s + 1) * d], (((1,), (1,)), ((), ())), preferred_element_type=_F32
    )
    sumsq = part if sumsq is None else sumsq + part
    proj = pp if proj is None else proj + pp
  inv = jax.lax.rsqrt(sumsq / (k * d) + cfg.rms_epsilon)
  raw = proj[:, 0:n] + proj[:, n : 2 * n] + proj[:, 2 * n : 3 * n]
  coef = coef_ref[...]
  gates = coef[2:3, :] * jax.nn.sigmoid(raw * inv * coef[0:1, :] + coef[1:2, :]) + cfg.hc_eps
  acc = gates[:, 0:1] * x_ref[:, 0:d].astype(_F32)
  for s in range(1, k):
    acc = acc + gates[:, s : s + 1] * x_ref[:, s * d : (s + 1) * d].astype(_F32)
  li_ref[...] = acc.astype(li_ref.dtype)
  gates_ref[...] = gates
  raw_ref[...] = raw
  inv_ref[...] = inv


def _post_fwd_kernel(x_ref, lo_ref, post_ref, out_ref, *, k, d):
  lo = lo_ref[...].astype(_F32)
  post = post_ref[...]
  for s in range(k):
    xs = x_ref[:, s * d : (s + 1) * d].astype(_F32)
    out_ref[:, s * d : (s + 1) * d] = (xs + post[:, s : s + 1] * lo).astype(out_ref.dtype)


def _post_bwd_kernel(g_ref, lo_ref, post_ref, dlo_ref, dpost_ref, *, k, d):
  """d_layer_output = sum_s post_s * g_s; d_post_s = sum_d g_s * layer_output. (d_x = g is passed through.)"""
  lo = lo_ref[...].astype(_F32)
  post = post_ref[...]
  lane = jax.lax.broadcasted_iota(jnp.int32, post.shape, 1)
  dlo, dpost = None, jnp.zeros(post.shape, _F32)
  for s in range(k):
    gs = g_ref[:, s * d : (s + 1) * d].astype(_F32)
    t = post[:, s : s + 1] * gs
    dlo = t if dlo is None else dlo + t
    dpost = jnp.where(lane == s, jnp.sum(gs * lo, axis=-1, keepdims=True), dpost)
  dlo_ref[...] = dlo.astype(dlo_ref.dtype)
  dpost_ref[...] = dpost


def _pre_bwd_kernel(
    x_ref,
    g_ref,
    dli_ref,
    dgate_ref,
    raw_ref,
    inv_ref,
    coef_ref,
    rhs_ref,
    pb0_ref,
    pb1_ref,
    pb2_ref,
    pa0_ref,
    pa1_ref,
    pa2_ref,
    dx_ref,
    dz_ref,
    dacc_ref,
    *,
    k,
    d,
    cfg,
):
  """d_x = g + pre_s * d_layer_input + (gate path); dz = d(gate pre-activation); d_alpha pieces accumulated."""

  @pl.when(pl.program_id(0) == 0)
  def _init():
    dacc_ref[...] = jnp.zeros(dacc_ref.shape, dacc_ref.dtype)

  raw = raw_ref[...]
  inv = inv_ref[...]
  coef = coef_ref[...]
  scale, beta, mult = coef[0:1, :], coef[1:2, :], coef[2:3, :]
  sg = jax.nn.sigmoid(raw * inv * scale + beta)
  pre = mult * sg + cfg.hc_eps  # lanes [0, k) are the pre gates
  dli = dli_ref[...].astype(_F32)

  # d_pre_s = sum_d x_s * d_layer_input; the cotangent of the post gates arrives in lanes [k, 2k) of dgate.
  dgate = dgate_ref[...]
  lane = jax.lax.broadcasted_iota(jnp.int32, dgate.shape, 1)
  for s in range(k):
    xs = x_ref[:, s * d : (s + 1) * d].astype(_F32)
    dgate = jnp.where(lane == s, jnp.sum(xs * dli, axis=-1, keepdims=True), dgate)

  dz = dgate * mult * sg * (1.0 - sg)
  dmix = dz * scale
  draw = dmix * inv
  dinv = jnp.sum(dmix * raw, axis=-1, keepdims=True)
  rms_coef = -dinv * inv * inv * inv / (k * d)  # d inv / d flat = -inv**3 * flat / (k*d)

  # Lay out the three bfloat16 pieces of draw for one-pass MXU contractions. Each placement matmul multiplies by 0/1
  # and adds zeros, so its float32 result is exactly the bfloat16 piece.
  pieces = [p.astype(_F32) for p in _split3(draw)]
  def place(refs):
    out = None
    for p, ref in zip(pieces, refs):
      t = jnp.dot(p, ref[...], preferred_element_type=_F32)
      out = t if out is None else out + t
    return out.astype(_BF16)

  lhs = place((pb0_ref, pb1_ref, pb2_ref))
  lhs_acc = place((pa0_ref, pa1_ref, pa2_ref))

  for s in range(k):
    sl = slice(s * d, (s + 1) * d)
    xs = x_ref[:, sl]
    xf = xs.astype(_F32)
    dflat = jnp.dot(lhs, rhs_ref[:, sl], preferred_element_type=_F32)
    dx = g_ref[:, sl].astype(_F32) + pre[:, s : s + 1] * dli + dflat + rms_coef * xf
    dx_ref[:, sl] = dx.astype(dx_ref.dtype)
    dacc_ref[:, sl] += jax.lax.dot_general(
        lhs_acc, xs.astype(_BF16), (((0,), (0,)), ((), ())), preferred_element_type=_F32
    )
  dz_ref[...] = dz


# ---------------------------------------------------------------------------------------------------------------------
# pallas_call builders on per-device (tokens, ...) arrays.
# ---------------------------------------------------------------------------------------------------------------------


def _pre_fwd_call(cfg, x2, rhs_fwd, coef, k):
  tokens, kd = x2.shape
  d, n, tb = kd // k, 2 * k, cfg.block_size
  _check_tokens(tokens, tb, "block_size")
  return pl.pallas_call(
      functools.partial(_pre_fwd_kernel, k=k, d=d, cfg=cfg),
      grid=(tokens // tb,),
      in_specs=[_rows(tb, kd), _whole(rhs_fwd.shape), _whole(coef.shape)],
      out_specs=[_rows(tb, d), _rows(tb, n), _rows(tb, n), _rows(tb, 1)],
      out_shape=[
          jax.ShapeDtypeStruct((tokens, d), x2.dtype),
          jax.ShapeDtypeStruct((tokens, n), _F32),
          jax.ShapeDtypeStruct((tokens, n), _F32),
          jax.ShapeDtypeStruct((tokens, 1), _F32),
      ],
      compiler_params=_compiler_params(cfg, pltpu.PARALLEL),
      cost_estimate=pl.CostEstimate(
          flops=2 * tokens * kd * _FWD_RHS_ROWS + 4 * tokens * kd,
          transcendentals=tokens * (n + 1),
          bytes_accessed=x2.nbytes + tokens * d * x2.dtype.itemsize,
      ),
      interpret=cfg.interpret,
  )(x2, rhs_fwd, coef)


def _post_fwd_call(cfg, x2, lo2, post2, k):
  tokens, kd = x2.shape
  d, tb = kd // k, cfg.block_size
  _check_tokens(tokens, tb, "block_size")
  return pl.pallas_call(
      functools.partial(_post_fwd_kernel, k=k, d=d),
      grid=(tokens // tb,),
      in_specs=[_rows(tb, kd), _rows(tb, d), _rows(tb, k)],
      out_specs=_rows(tb, kd),
      out_shape=jax.ShapeDtypeStruct((tokens, kd), x2.dtype),
      compiler_params=_compiler_params(cfg, pltpu.PARALLEL),
      cost_estimate=pl.CostEstimate(flops=2 * tokens * kd, transcendentals=0, bytes_accessed=2 * x2.nbytes + lo2.nbytes),
      interpret=cfg.interpret,
  )(x2, lo2, post2)


def _post_bwd_call(cfg, g2, lo2, post2, k):
  tokens, kd = g2.shape
  d, tb = kd // k, cfg.bwd_block_size
  _check_tokens(tokens, tb, "bwd_block_size")
  return pl.pallas_call(
      functools.partial(_post_bwd_kernel, k=k, d=d),
      grid=(tokens // tb,),
      in_specs=[_rows(tb, kd), _rows(tb, d), _rows(tb, k)],
      out_specs=[_rows(tb, d), _rows(tb, k)],
      out_shape=[jax.ShapeDtypeStruct((tokens, d), lo2.dtype), jax.ShapeDtypeStruct((tokens, k), _F32)],
      compiler_params=_compiler_params(cfg, pltpu.PARALLEL),
      cost_estimate=pl.CostEstimate(flops=4 * tokens * kd, transcendentals=0, bytes_accessed=g2.nbytes + 2 * lo2.nbytes),
      interpret=cfg.interpret,
  )(g2, lo2, post2)


def _pre_bwd_call(cfg, x2, g2, dli2, dgate2, raw2, inv2, coef, rhs_bwd, k):
  tokens, kd = x2.shape
  d, n, tb = kd // k, 2 * k, cfg.bwd_block_size
  _check_tokens(tokens, tb, "bwd_block_size")
  place_bwd, place_acc = _placements(n)
  small = [_whole(p.shape) for p in place_bwd + place_acc]
  return pl.pallas_call(
      functools.partial(_pre_bwd_kernel, k=k, d=d, cfg=cfg),
      grid=(tokens // tb,),
      in_specs=[
          _rows(tb, kd),
          _rows(tb, kd),
          _rows(tb, d),
          _rows(tb, n),
          _rows(tb, n),
          _rows(tb, 1),
          _whole(coef.shape),
          _whole(rhs_bwd.shape),
          *small,
      ],
      out_specs=[_rows(tb, kd), _rows(tb, n), _whole((_FWD_RHS_ROWS, kd))],
      out_shape=[
          jax.ShapeDtypeStruct((tokens, kd), x2.dtype),
          jax.ShapeDtypeStruct((tokens, n), _F32),
          jax.ShapeDtypeStruct((_FWD_RHS_ROWS, kd), _F32),
      ],
      # The d_alpha accumulator is revisited by every grid step.
      compiler_params=_compiler_params(cfg, pltpu.ARBITRARY),
      cost_estimate=pl.CostEstimate(
          flops=2 * tokens * kd * (_BWD_RHS_ROWS + _FWD_RHS_ROWS) + 8 * tokens * kd,
          transcendentals=tokens * n,
          bytes_accessed=3 * x2.nbytes + dli2.nbytes,
      ),
      interpret=cfg.interpret,
  )(x2, g2, dli2, dgate2, raw2, inv2, coef, rhs_bwd, *place_bwd, *place_acc)


# ---------------------------------------------------------------------------------------------------------------------
# Sharding: kernels are token-local, so each device runs them on its own tokens.
# ---------------------------------------------------------------------------------------------------------------------


def _token_axis_names(cfg):
  names = []
  for entry in cfg.token_axes:
    if entry is None:
      continue
    names.extend(entry if isinstance(entry, tuple) else (entry,))
  return tuple(names)


def _sharded(cfg, fn, in_kinds, out_kinds):
  """Runs fn on per-device token rows. Kinds: 't' = (batch, seq, ...) token array, 'r' = replicated, 'sum' = output
  summed over all token shards. Output kinds come with their rank: (kind, ndim).

  A 'sum' output leaves the shard_map with one partial per token shard and is summed outside it, so XLA plans the
  cross-device reduction like any replicated-parameter gradient (rather than an explicit psum inside every layer's
  backward, which on multi-slice meshes would cross DCN once per layer)."""
  b_ax, s_ax = cfg.token_axes
  names = _token_axis_names(cfg)

  def spec(kind, ndim):
    if kind == "t":
      return P(b_ax, s_ax, *([None] * (ndim - 2)))
    if kind == "sum":
      return P(names or None, *([None] * ndim))
    return P(*([None] * ndim))

  if cfg.mesh is None:
    return fn
  single = len(out_kinds) == 1
  kinds = [kind for kind, _ in out_kinds]
  out_specs = tuple(spec(kind, ndim) for kind, ndim in out_kinds)

  def local(*args):
    outs = fn(*args)
    outs = (outs,) if single else tuple(outs)
    outs = tuple(o[None] if kind == "sum" else o for o, kind in zip(outs, kinds))
    return outs[0] if single else outs

  def wrapped(*args):
    in_specs = tuple(spec(kind, a.ndim) for kind, a in zip(in_kinds, args))
    outs = jax.shard_map(
        local, mesh=cfg.mesh, in_specs=in_specs, out_specs=out_specs[0] if single else out_specs, check_vma=False
    )(*args)
    outs = (outs,) if single else tuple(outs)
    outs = tuple(jnp.sum(o, axis=0) if kind == "sum" else o for o, kind in zip(outs, kinds))
    return outs[0] if single else outs

  return wrapped


def _flat(a):
  """(batch, seq, [k,] w) -> (batch*seq, [k*] w)."""
  return a.reshape(a.shape[0] * a.shape[1], -1)


def _pre_fwd(cfg, x, alpha, scale, beta):
  k, d = x.shape[2], x.shape[3]
  rhs_fwd, _ = _gate_rhs(alpha)
  coef = _coef(scale, beta, k, cfg.post_magnitude)

  def local(x, rhs_fwd, coef):
    bl, sl = x.shape[:2]
    li, gates, raw, inv = _pre_fwd_call(cfg, _flat(x), rhs_fwd, coef, k)
    return li.reshape(bl, sl, d), gates.reshape(bl, sl, 2 * k), raw.reshape(bl, sl, 2 * k), inv.reshape(bl, sl, 1)

  li, gates, raw, inv = _sharded(cfg, local, ("t", "r", "r"), (("t", 3), ("t", 3), ("t", 3), ("t", 3)))(x, rhs_fwd, coef)
  return li, gates[..., k:], raw, inv


def _validate(x, alpha):
  if x.ndim != 4:
    raise ValueError(f"iHC kernel expects x of shape (batch, seq, streams, emb); got {x.shape}.")
  if x.dtype != _BF16:
    # The one-pass gate projection relies on x being exact in bfloat16.
    raise ValueError(f"iHC kernel requires bfloat16 activations; got {x.dtype}.")
  k, d = x.shape[2], x.shape[3]
  if d % 128:
    raise ValueError(f"iHC kernel requires the embedding dim to be a multiple of 128; got {d}.")
  if k > MAX_STREAMS or alpha.shape != (k * d, 2 * k):
    raise ValueError(f"iHC kernel: needs at most {MAX_STREAMS} streams and alpha of shape ({k * d}, {2 * k}); got k={k}, {alpha.shape}.")


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def ihc_pre(cfg: IhcKernelConfig, x, alpha, scale, beta):
  """Gates and branch input of an iHC block.

  Args:
    cfg: kernel configuration.
    x: (batch, seq, k, d) bfloat16 streams.
    alpha: (k*d, 2k) float32 gate projection [pre_alpha, post_alpha].
    scale: (2k,) float32 gate scales (pre_alpha_scale repeated k times, then post_alpha_scale).
    beta: (2k,) float32 gate biases [pre_beta, post_beta].

  Returns:
    (layer_input (batch, seq, d), x unchanged, post gates (batch, seq, k) float32); pass the last two to `ihc_post`.
  """
  _validate(x, alpha)
  li, post, _, _ = _pre_fwd(cfg, x, alpha, scale, beta)
  return li, x, post


def _ihc_pre_fwd(cfg, x, alpha, scale, beta):
  _validate(x, alpha)
  li, post, raw, inv = _pre_fwd(cfg, x, alpha, scale, beta)
  return (li, x, post), (x, alpha, scale, beta, raw, inv)


def _ihc_pre_bwd(cfg, res, cts):
  x, alpha, scale, beta, raw, inv = res
  d_li, d_x_post, d_post = cts
  k, d = x.shape[2], x.shape[3]
  n = 2 * k
  _, rhs_bwd = _gate_rhs(alpha)
  coef = _coef(scale, beta, k, cfg.post_magnitude)
  dgate = jnp.concatenate([jnp.zeros_like(d_post, _F32), d_post.astype(_F32)], axis=-1)

  def local(x, g, dli, dgate, raw, inv, coef, rhs_bwd):
    bl, sl = x.shape[:2]
    dx, dz, dacc = _pre_bwd_call(cfg, _flat(x), _flat(g), _flat(dli), _flat(dgate), _flat(raw), _flat(inv), coef, rhs_bwd, k)
    # Sum the three pieces before the cross-device sum: d_alpha^T is (n, k*d).
    dalpha_t = dacc[0:n] + dacc[n : 2 * n] + dacc[2 * n : 3 * n]
    return dx.reshape(bl, sl, k, d), dz.reshape(bl, sl, n), dalpha_t

  dx, dz, dalpha_t = _sharded(
      cfg, local, ("t", "t", "t", "t", "t", "t", "r", "r"), (("t", 4), ("t", 3), ("sum", 2))
  )(x, d_x_post.astype(x.dtype), d_li.astype(x.dtype), dgate, raw, inv, coef, rhs_bwd)
  mixes = raw * inv
  d_scale = jnp.sum(dz * mixes, axis=(0, 1))
  d_beta = jnp.sum(dz, axis=(0, 1))
  return dx, dalpha_t.T.astype(alpha.dtype), d_scale.astype(scale.dtype), d_beta.astype(beta.dtype)


ihc_pre.defvjp(_ihc_pre_fwd, _ihc_pre_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def ihc_post(cfg: IhcKernelConfig, layer_output, x, post):
  """Residual update of an iHC block: out_s = x_s + post_s * layer_output, in float32, cast to x.dtype.

  Args:
    cfg: kernel configuration.
    layer_output: (batch, seq, d) branch output.
    x: the x returned by `ihc_pre`.
    post: (batch, seq, k) float32 post gates returned by `ihc_pre`.
  """
  return _post_fwd(cfg, layer_output, x, post)


def _post_fwd(cfg, layer_output, x, post):
  k = x.shape[2]

  def local(x, lo, post):
    out = _post_fwd_call(cfg, _flat(x), _flat(lo), _flat(post), k)
    return out.reshape(x.shape)

  return _sharded(cfg, local, ("t", "t", "t"), (("t", 4),))(x, layer_output, post)


def _ihc_post_fwd(cfg, layer_output, x, post):
  return _post_fwd(cfg, layer_output, x, post), (layer_output, post)


def _ihc_post_bwd(cfg, res, g):
  layer_output, post = res
  k = post.shape[-1]

  def local(g, lo, post):
    bl, sl = g.shape[:2]
    dlo, dpost = _post_bwd_call(cfg, _flat(g), _flat(lo), _flat(post), k)
    return dlo.reshape(lo.shape), dpost.reshape(bl, sl, k)

  dlo, dpost = _sharded(cfg, local, ("t", "t", "t"), (("t", 3), ("t", 3)))(g, layer_output, post)
  # d_x = g: the identity residual. It reaches ihc_pre as the cotangent of its x output.
  return dlo, g, dpost


ihc_post.defvjp(_ihc_post_fwd, _ihc_post_bwd)
