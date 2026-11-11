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

"""splash_static_causal_partial_mask: static causal schedule + indexer mask tiles vs make_dynamic_splash_mha.

The kernels run in Pallas interpret mode on CPU. In that mode an aliased input does not see earlier grid steps'
writes to its output (checked with a two-block probe on jax 0.11.1), so the dQ ring buffer that the dynamic mode
forces (dq_reduction_steps=3) accumulates wrongly there: dQ of the dynamic path cannot be the CPU reference. dQ of
the static path is therefore checked against a dense reference, and against the dynamic path only when both use the
same ring buffer.
"""

import dataclasses
import functools
import sys
import types
import unittest
from unittest import mock

from absl.testing import parameterized
import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import base as tokamax_base
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as tokamax_mask_info
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk

from maxtext.common.common_types import AttentionType, DEFAULT_MASK_VALUE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.layers import attention_op
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path

B, H, S, D, BLK, TOPK = 3, 2, 512, 128, 128, 128


# dKV blocks of 256 (2 kv steps) keep the static path on its per-kv-block dQ buffer, which interpret mode computes
# correctly; past 3 kv steps it switches to the dQ ring (see test_dq_ring_past_three_kv_blocks).
DKV_BLK = 256


def _config(layout="HEAD_DIM_MINOR", dq_reduction_steps=None, dkv_block=DKV_BLK):
  layout = sk.QKVLayout[layout]
  return sk.SplashConfig(
      block_q=BLK,
      block_kv=BLK,
      block_kv_compute=BLK,
      block_q_dkv=dkv_block,
      block_kv_dkv=dkv_block,
      block_kv_dkv_compute=dkv_block,
      use_fused_bwd_kernel=True,
      q_layout=layout,
      k_layout=layout,
      v_layout=layout,
      interpret=True,
      residual_checkpoint_name="context",
      dq_reduction_steps=dq_reduction_steps,
  )


def _topk_causal_mask(key, segment_ids=None):
  """Causal top-TOPK selection on random scores, like the MLA indexer (causal applied before and after top-k)."""
  allowed = jnp.tril(jnp.ones((S, S), bool))[None]
  if segment_ids is not None:
    allowed = allowed & (segment_ids[:, :, None] == segment_ids[:, None, :])
  scores = jnp.where(allowed, jax.random.normal(key, (B, S, S)), -1e30)
  kth = jax.lax.top_k(scores, TOPK)[0][..., -1:]
  return (scores >= kth) & allowed


def _inputs(case, dtype=jnp.float32):
  """q, k, v [B, H, S, D], segment ids, sinks [H], bool mask [B, S, S] and an output cotangent."""
  ks = jax.random.split(jax.random.PRNGKey(0), 6)
  q, k, v, ct = (jax.random.normal(key, (B, H, S, D), dtype) for key in (ks[0], ks[1], ks[2], ks[5]))
  sinks = jax.random.normal(ks[3], (H,), jnp.float32)
  segment_ids = None
  if case == "packed":
    ids = jnp.asarray(np.repeat(np.arange(1, 5), [100, 200, 72, 140]))
    segment_ids = sk.SegmentIds(jnp.broadcast_to(ids, (B, S)), jnp.broadcast_to(ids, (B, S)))
  # The packed mask deliberately ignores segments, so the result depends on folding the segment ids into the tiles.
  mask = _topk_causal_mask(ks[4])
  if case == "empty_block":
    # Visited causal blocks with no selected element, different per batch element.
    mask = mask.at[1, 2 * BLK : 3 * BLK, BLK : 2 * BLK].set(False)
    mask = mask.at[2, 3 * BLK : 4 * BLK, 3 * BLK : 4 * BLK].set(False)
  return q, k, v, segment_ids, sinks, mask, ct


def _dynamic(q, k, v, segment_ids, sinks, mask, *, config, save_residuals):
  """The splash_static_causal_partial_mask=False path of attention_op (d45d4e60)."""

  def one(q, k, v, segment_ids, sinks, mask):
    kernel = sk.make_dynamic_splash_mha(mask=mask, config=config)
    return kernel(q, k, v, segment_ids, sinks=sinks, save_residuals=save_residuals)

  return jax.vmap(one, (0, 0, 0, 0, None, 0))(q, k, v, segment_ids, sinks, mask)


def _static(q, k, v, segment_ids, sinks, mask, *, config, save_residuals):
  fn = functools.partial(attention_op.static_causal_partial_mask_splash, config=config, save_residuals=save_residuals)
  return jax.vmap(fn, (0, 0, 0, 0, None, 0))(q, k, v, segment_ids, sinks, mask)


def _dense(q, k, v, segment_ids, sinks, mask, **_):
  if segment_ids is not None:
    mask = mask & (segment_ids.q[:, :, None] == segment_ids.kv[:, None, :])
  logits = jnp.einsum("bhsd,bhtd->bhst", q, k, precision="highest")
  logits = jnp.where(mask[:, None], logits, tokamax_base.DEFAULT_MASK_VALUE)
  sink = sinks[None, :, None, None]
  m = jnp.maximum(logits.max(-1, keepdims=True), sink)
  p = jnp.exp(logits - m)
  p = p / (p.sum(-1, keepdims=True) + jnp.exp(sink - m))
  return jnp.einsum("bhst,bhtd->bhsd", p, v, precision="highest")


def _forward_and_grads(fn, q, k, v, segment_ids, sinks, mask, ct, config):
  out = None
  if fn in (_dynamic, _static):
    out = jax.jit(functools.partial(fn, config=config, save_residuals=True))(q, k, v, segment_ids, sinks, mask)

  def loss(q, k, v, sinks):
    return jnp.sum(fn(q, k, v, segment_ids, sinks, mask, config=config, save_residuals=False).astype(jnp.float32) * ct)

  return out, jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(q, k, v, sinks), loss


def _loop_counts(jaxpr, in_loop=False, counts=None):
  """Counts pallas_calls inside loops (vmap's explicit-loop fallback is a fori_loop) and dynamic slices."""
  counts = counts if counts is not None else {"pallas_in_loop": 0, "pallas": 0, "dynamic_slice": 0}
  for eqn in jaxpr.eqns:
    name = eqn.primitive.name
    if name == "pallas_call":
      counts["pallas"] += 1
      counts["pallas_in_loop"] += int(in_loop)
      continue
    counts["dynamic_slice"] += int(name in ("dynamic_slice", "dynamic_update_slice"))
    for param in eqn.params.values():
      for sub in param if isinstance(param, (list, tuple)) else (param,):
        sub = getattr(sub, "jaxpr", sub)
        if hasattr(sub, "eqns"):
          _loop_counts(sub, in_loop or name in ("while", "scan"), counts)
  return counts


class StaticCausalPartialMaskKernelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("plain", "plain", "HEAD_DIM_MINOR"),
      ("packed_segments", "packed", "HEAD_DIM_MINOR"),
      ("empty_visited_blocks", "empty_block", "HEAD_DIM_MINOR"),
      ("seq_minor", "plain", "SEQ_MINOR"),
  )
  def test_matches_dynamic_splash(self, case, layout):
    q, k, v, segment_ids, sinks, mask, ct = _inputs(case)
    config = _config(layout)
    dyn_out, dyn_grads, _ = _forward_and_grads(_dynamic, q, k, v, segment_ids, sinks, mask, ct, config)
    sta_out, sta_grads, _ = _forward_and_grads(_static, q, k, v, segment_ids, sinks, mask, ct, config)
    _, ref_grads, _ = _forward_and_grads(_dense, q, k, v, segment_ids, sinks, mask, ct, config)

    # Bitwise: same per-block arithmetic in the same order; a visited all-masked block adds exact zeros. This holds
    # for every row only because sinks are present: in the packed case some rows keep no key after the segment fold
    # (the mask ignores segments), and with sinks such a row is 0 in both paths (see test_rows_without_keys).
    np.testing.assert_array_equal(sta_out[0], dyn_out[0])
    np.testing.assert_array_equal(sta_out[1]["logsumexp"], dyn_out[1]["logsumexp"])
    np.testing.assert_array_equal(sta_out[1]["max_logits"], dyn_out[1]["max_logits"])
    for name, i in (("dk", 1), ("dv", 2), ("dsinks", 3)):
      np.testing.assert_array_equal(sta_grads[i], dyn_grads[i], err_msg=name)
    # dQ: see the module docstring. fp32 kernel vs HIGHEST dense reference.
    np.testing.assert_allclose(sta_grads[0], ref_grads[0], rtol=0, atol=2e-3)
    np.testing.assert_allclose(sta_grads[1], ref_grads[1], rtol=0, atol=2e-3)

  def test_gqa(self):
    """One kv head (absorbed-MQA layout). The dynamic GQA backward accumulates dK/dV through aliased outputs, which
    interpret mode breaks the same way as the dQ ring, so the gradients are checked against the dense reference."""
    q, k, v, segment_ids, sinks, mask, ct = _inputs("empty_block")
    k, v = k[:, :1], v[:, :1]
    config = _config()
    dyn_out, _, _ = _forward_and_grads(_dynamic, q, k, v, segment_ids, sinks, mask, ct, config)
    sta_out, sta_grads, _ = _forward_and_grads(_static, q, k, v, segment_ids, sinks, mask, ct, config)

    def dense(q, k, v, *args, **_):
      return _dense(q, jnp.repeat(k, H, 1), jnp.repeat(v, H, 1), *args)

    _, ref_grads, _ = _forward_and_grads(dense, q, k, v, segment_ids, sinks, mask, ct, config)
    np.testing.assert_array_equal(sta_out[0], dyn_out[0])
    np.testing.assert_array_equal(sta_out[1]["logsumexp"], dyn_out[1]["logsumexp"])
    for name, i in (("dq", 0), ("dk", 1), ("dv", 2), ("dsinks", 3)):
      np.testing.assert_allclose(sta_grads[i], ref_grads[i], rtol=0, atol=2e-3, err_msg=name)

  def test_rows_without_keys(self):
    """Without sinks, rows that keep no key after the segment fold differ (mean of V over different visited blocks);
    every other row, and the gradients when those rows get no cotangent, still match bitwise."""
    q, k, v, _, _, mask, ct = _inputs("plain")
    # The last segment starts inside q block 3, whose segment-id range then misses kv block 0: the dynamic schedule
    # prunes that block for q block 3 while the static one visits it, so a keyless row there sees different blocks.
    ids = jnp.asarray(np.repeat(np.arange(1, 5), [100, 200, 120, 92]))
    segment_ids = sk.SegmentIds(jnp.broadcast_to(ids, (B, S)), jnp.broadcast_to(ids, (B, S)))
    config = _config()
    folded = mask & (segment_ids.q[:, :, None] == segment_ids.kv[:, None, :])
    empty_rows = ~folded.any(-1)  # [B, S]
    self.assertTrue(bool(empty_rows.any()))  # the case under test does occur
    ct = jnp.where(empty_rows[:, None, :, None], 0.0, ct)
    results = []
    for fn in (_dynamic, _static):
      out, stats = jax.jit(functools.partial(fn, config=config, save_residuals=True))(q, k, v, segment_ids, None, mask)

      def loss(q, k, v, fn=fn):
        return jnp.sum(fn(q, k, v, segment_ids, None, mask, config=config, save_residuals=False) * ct)

      results.append((out, stats["logsumexp"], jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)))
    dyn_out, dyn_lse, dyn_grads = results[0]
    sta_out, sta_lse, sta_grads = results[1]
    kept = ~empty_rows[:, None, :]
    np.testing.assert_array_equal(np.where(kept[..., None], sta_out, 0), np.where(kept[..., None], dyn_out, 0))
    np.testing.assert_array_equal(np.where(kept, sta_lse, 0), np.where(kept, dyn_lse, 0))
    self.assertFalse(np.array_equal(np.where(kept[..., None], 0, sta_out), np.where(kept[..., None], 0, dyn_out)))
    for name, i in (("dk", 1), ("dv", 2)):
      np.testing.assert_array_equal(sta_grads[i], dyn_grads[i], err_msg=name)

  def test_cost_estimate_uses_the_causal_block_fraction(self):
    q, k, v, segment_ids, sinks, mask, _ = _inputs("plain")
    with mock.patch.object(sk, "SplashAttentionKernel", side_effect=RuntimeError("captured")) as kernel:
      with self.assertRaisesRegex(RuntimeError, "captured"):
        attention_op.static_causal_partial_mask_splash(q[0], k[0], v[0], segment_ids, sinks, mask[0], config=_config())
    causal_fraction = lambda n: n * (n + 1) / 2 / n**2  # pylint: disable=unnecessary-lambda-assignment
    self.assertEqual(kernel.call_args.kwargs["fwd_mask_sparsity"], causal_fraction(S // BLK))
    self.assertEqual(kernel.call_args.kwargs["dkv_mask_sparsity"], causal_fraction(S // DKV_BLK))

  def test_dq_ring_past_three_kv_blocks(self):
    """With dq_reduction_steps unset and more than 3 dKV kv blocks, the static path bounds its dQ buffer with the
    same 3-slot ring as the dynamic path; with at most 3 it keeps one slice per kv block."""
    q, k, v, segment_ids, sinks, mask, ct = _inputs("plain")
    for dkv_block, expected in ((BLK, 3), (DKV_BLK, None)):
      with self.subTest(dkv_block=dkv_block):
        with mock.patch.object(sk, "SplashAttentionKernel", side_effect=RuntimeError("captured")) as kernel:
          with self.assertRaisesRegex(RuntimeError, "captured"):
            attention_op.static_causal_partial_mask_splash(
                q[0], k[0], v[0], segment_ids, sinks, mask[0], config=_config(dkv_block=dkv_block)
            )
        self.assertEqual(kernel.call_args.kwargs["config"].dq_reduction_steps, expected)
        self.assertEqual(kernel.call_args.args[1].active_rows is not None, expected == 3)  # compact dKV grid
    # dQ through the ring equals the dynamic path's (same ring, same visit order: no empty block in "plain").
    config = _config(dkv_block=BLK)
    _, dyn_grads, _ = _forward_and_grads(_dynamic, q, k, v, segment_ids, sinks, mask, ct, config)
    _, sta_grads, _ = _forward_and_grads(_static, q, k, v, segment_ids, sinks, mask, ct, config)
    for i in range(4):
      np.testing.assert_array_equal(sta_grads[i], dyn_grads[i])

  def test_dq_matches_dynamic_with_the_same_reduction(self):
    """With dq_reduction_steps=3 both paths reduce dQ through the same ring buffer, so the per-block dQ agree.

    Needs a mask with no empty causal block: in interpret mode the (broken) ring result depends on the exact sequence
    of visited blocks, which differs once the dynamic schedule skips a block.
    """
    q, k, v, segment_ids, sinks, mask, ct = _inputs("plain")
    config = _config(dq_reduction_steps=3)
    _, dyn_grads, _ = _forward_and_grads(_dynamic, q, k, v, segment_ids, sinks, mask, ct, config)
    _, sta_grads, _ = _forward_and_grads(_static, q, k, v, segment_ids, sinks, mask, ct, config)
    for i in range(4):
      np.testing.assert_array_equal(sta_grads[i], dyn_grads[i])

  def test_static_schedule_removes_the_per_batch_loop(self):
    q, k, v, segment_ids, sinks, mask, ct = _inputs("packed")
    counts = {}
    for name, fn in (("dynamic", _dynamic), ("static", _static)):
      _, _, loss = _forward_and_grads(fn, q, k, v, segment_ids, sinks, mask, ct, _config())
      counts[name] = _loop_counts(jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3)))(q, k, v, sinks).jaxpr)
    # Positive control: the dynamic path hits pallas_call's explicit-loop batching (forward and backward kernels).
    self.assertEqual(counts["dynamic"]["pallas_in_loop"], 2, counts)
    self.assertGreater(counts["dynamic"]["dynamic_slice"], 0, counts)
    self.assertEqual(counts["static"], {"pallas_in_loop": 0, "pallas": 2, "dynamic_slice": 0})

  def test_schedule_visits_exactly_the_causal_blocks(self):
    fwd, sparsity = attention_op._static_causal_schedule(S, S, BLK, BLK, False, True)  # pylint: disable=protected-access
    n = S // BLK
    expected = [(i, j) for i in range(n) for j in range(n) if j <= i]
    self.assertEqual(list(zip(fwd.active_rows.tolist(), fwd.active_cols.tolist())), expected)
    self.assertEqual(fwd.num_active_blocks.tolist(), [len(expected)])
    self.assertEqual(fwd.mask_next.tolist(), [i * n + j for i, j in expected])
    np.testing.assert_array_equal(fwd.block_mask, 1)
    self.assertEqual(sparsity, len(expected) / n**2)
    dkv, _ = attention_op._static_causal_schedule(S, S, BLK, BLK, True, False)  # pylint: disable=protected-access
    self.assertIsNone(dkv.active_rows)
    # Static dKV grid walks (kv block j, q block i) and runs block (i, j) iff j <= i.
    np.testing.assert_array_equal(dkv.block_mask.reshape(n, n), np.triu(np.ones((n, n), np.int8)))

  def test_rejects_diagonal_skip(self):
    q, k, v, segment_ids, sinks, mask, _ = _inputs("plain")
    config = dataclasses.replace(_config(), qk_diag_skip=True)
    with self.assertRaisesRegex(ValueError, "diag_skip"):
      attention_op.static_causal_partial_mask_splash(q[0], k[0], v[0], segment_ids, sinks, mask[0], config=config)


def _make_op(flag, attention_type=AttentionType.MLA, **overrides):
  """A real AttentionOp (flash, tokamax splash, 128 blocks) on the 1-device CPU mesh."""
  args = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": S,
      "attention": "flash",
      "use_tokamax_splash": True,
      "splash_static_causal_partial_mask": flag,
      "use_indexer": True,
      "attention_type": "mla",
      "q_lora_rank": 64,
      "indexer_topk": TOPK,
      **{f"sa_block_{name}": BLK for name in ("q", "kv", "kv_compute", "q_dq", "kv_dq")},
      **{f"sa_block_{name}": DKV_BLK for name in ("q_dkv", "kv_dkv", "kv_dkv_compute")},
      **overrides,
  }
  cfg = pyconfig.initialize([sys.argv[0], get_test_config_path()], **args)
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  return attention_op.AttentionOp(
      config=cfg,
      num_query_heads=H,
      num_kv_heads=H,
      max_target_length=S,
      mesh=mesh,
      attention_kernel="flash",
      attention_type=attention_type,
  )


_REAL_SPLASH_CONFIG = sk.SplashConfig


class StaticCausalPartialMaskOpTest(unittest.TestCase):
  """The flag through AttentionOp.tpu_flash_attention (1-device CPU mesh, kernels in interpret mode)."""

  def setUp(self):
    """Inputs in the op layout [b, S, H, D]; two packed segments; a causal random selection as indexer mask."""
    super().setUp()
    ks = jax.random.split(jax.random.PRNGKey(1), 6)
    self.b = 2
    shape = (self.b, S, H, D)
    self.q, self.k, self.v, self.ct = (jax.random.normal(key, shape) for key in (ks[0], ks[1], ks[2], ks[5]))
    self.sinks = jax.random.normal(ks[3], (H,))
    self.segment_ids = jnp.asarray(np.broadcast_to(np.repeat([1, 2], [200, S - 200]), (self.b, S)))
    allowed = jnp.tril(jnp.ones((S, S), bool))[None] & (self.segment_ids[:, :, None] == self.segment_ids[:, None, :])
    selected = (jax.random.uniform(ks[4], (self.b, S, S)) < 0.3) | jnp.eye(S, dtype=bool)[None]
    # Additive mask, as the MLA indexer hands it to the attention op.
    self.indexer_mask = jnp.where(allowed & selected, 0.0, DEFAULT_MASK_VALUE)

  def _run(self, op, indexer_mask=None):
    """Output, (dq, dk, dv, dsinks) and loop counts of the grad jaxpr for one op, kernels in interpret mode."""
    indexer_mask = self.indexer_mask if indexer_mask is None else indexer_mask

    def attend(q, k, v, sinks):
      return op.tpu_flash_attention(q, k, v, self.segment_ids, sinks=sinks, indexer_mask=indexer_mask)[0]

    def loss(q, k, v, sinks):
      return jnp.sum(attend(q, k, v, sinks) * self.ct)

    with mock.patch.object(sk, "SplashConfig", functools.partial(_REAL_SPLASH_CONFIG, interpret=True)):
      out = jax.jit(attend)(self.q, self.k, self.v, self.sinks)
      grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(self.q, self.k, self.v, self.sinks)
      counts = _loop_counts(
          jax.make_jaxpr(jax.grad(loss, argnums=(0, 1, 2, 3)))(self.q, self.k, self.v, self.sinks).jaxpr
      )
    return out, grads, counts

  def test_flag_matches_dynamic_path_without_the_loop(self):
    off_out, off_grads, off_counts = self._run(_make_op(False))
    on_out, on_grads, on_counts = self._run(_make_op(True))
    np.testing.assert_array_equal(on_out, off_out)
    for i in (1, 2, 3):  # dk, dv, dsinks; dq of the dynamic path is not valid in interpret mode (module docstring).
      np.testing.assert_array_equal(on_grads[i], off_grads[i])
    mask = self.indexer_mask == 0.0

    def to_bhsd(x):
      return jnp.transpose(x, (0, 2, 1, 3))

    def ref_loss(q, k, v, sinks):
      out = _dense(to_bhsd(q), to_bhsd(k), to_bhsd(v), None, sinks, mask)
      return jnp.sum(to_bhsd(out) * self.ct)

    ref_dq = jax.grad(ref_loss)(self.q, self.k, self.v, self.sinks)
    np.testing.assert_allclose(on_grads[0], ref_dq, rtol=0, atol=2e-3)
    self.assertEqual(off_counts["pallas_in_loop"], 2, off_counts)
    self.assertEqual((on_counts["pallas_in_loop"], on_counts["dynamic_slice"]), (0, 0), on_counts)

  def test_record_max_logits(self):
    stats = []
    for flag in (False, True):
      op = _make_op(flag)
      with mock.patch.object(sk, "SplashConfig", functools.partial(_REAL_SPLASH_CONFIG, interpret=True)):
        stats.append(
            jax.jit(
                lambda q, k, v, op=op: op.tpu_flash_attention(
                    q, k, v, self.segment_ids, sinks=self.sinks, indexer_mask=self.indexer_mask, record_max_logits=True
                )
            )(self.q, self.k, self.v)
        )
    self.assertIsNotNone(stats[0][1])
    np.testing.assert_array_equal(stats[1][0], stats[0][0])
    np.testing.assert_array_equal(stats[1][1], stats[0][1])

  _COMMON = {
      "model_mode": MODEL_MODE_TRAIN,
      "previous_chunk": None,
      "bidirectional_mask": None,
      "cp_size": 1,
      "context_parallel_kernel": False,
      "q_len": S,
      "kv_len": S,
      "has_sinks": True,
  }

  def _decide(self, op, sa_config=None, **kwargs):
    # pylint: disable-next=protected-access
    return op._use_splash_static_causal_partial_mask(sa_config=sa_config or _config(), **{**self._COMMON, **kwargs})

  def test_unsupported_calls_keep_the_dynamic_path(self):
    op = _make_op(True)
    self.assertTrue(self._decide(op))
    cases = (
        ({}, {"cp_size": 2}),
        ({}, {"context_parallel_kernel": True}),
        ({}, {"model_mode": MODEL_MODE_PREFILL}),
        ({}, {"previous_chunk": jnp.zeros((1, 8))}),
        ({"attention_type": AttentionType.FULL}, {}),
        ({"attention_type": AttentionType.LOCAL_SLIDING}, {}),
        ({"attention_type": AttentionType.CHUNK}, {}),
        ({}, {"bidirectional_mask": jnp.ones((1, S), bool)}),
        ({}, {"kv_len": S + BLK}),
        ({}, {"q_len": S + 64, "kv_len": S + 64}),
        ({}, {"sa_config": dataclasses.replace(_config(), qk_diag_skip=True)}),
    )
    for op_kwargs, call_kwargs in cases:
      with self.subTest(**{k: str(v) for k, v in {**op_kwargs, **call_kwargs}.items()}):
        self.assertFalse(self._decide(_make_op(True, **op_kwargs) if op_kwargs else op, **call_kwargs))
    with mock.patch.object(op, "config", types.SimpleNamespace(**{**dict(op.config.get_keys()), "use_multimodal": True})):
      self.assertFalse(self._decide(op))

  def test_warns_once_without_sinks(self):
    op = _make_op(True)
    attention_op._log_once.cache_clear()  # pylint: disable=protected-access
    with mock.patch.object(attention_op.max_logging, "warning") as warning:
      self.assertTrue(self._decide(op))
      warning.assert_not_called()
      for _ in range(2):
        self.assertTrue(self._decide(op, has_sinks=False))
      warning.assert_called_once()
      self.assertIn("sinks", warning.call_args.args[0])

  def _dynamic_forward_counts(self, op, **kwargs):
    def attend(q, k, v):
      return op.tpu_flash_attention(q, k, v, self.segment_ids, sinks=self.sinks, indexer_mask=self.indexer_mask, **kwargs)

    with mock.patch.object(sk, "SplashConfig", functools.partial(_REAL_SPLASH_CONFIG, interpret=True)):
      return jax.jit(attend)(self.q, self.k, self.v)[0], _loop_counts(
          jax.make_jaxpr(attend)(self.q, self.k, self.v).jaxpr
      )

  def test_prefill_falls_back_through_tpu_flash_attention(self):
    off_out, off_counts = self._dynamic_forward_counts(_make_op(False), model_mode=MODEL_MODE_PREFILL)
    on_out, on_counts = self._dynamic_forward_counts(_make_op(True), model_mode=MODEL_MODE_PREFILL)
    np.testing.assert_array_equal(on_out, off_out)
    self.assertEqual(on_counts, off_counts)
    self.assertEqual(on_counts["pallas_in_loop"], 1)  # still the dynamic kernel under the explicit loop

  def test_non_causal_attention_falls_back_through_tpu_flash_attention(self):
    _, counts = self._dynamic_forward_counts(_make_op(True, attention_type=AttentionType.FULL))
    self.assertEqual(counts["pallas_in_loop"], 1)

  def test_local_shapes_match(self):
    q = jnp.zeros((1, H, S, D))
    mask = jnp.zeros((1, S, S), bool)
    self.assertTrue(attention_op.static_causal_local_shapes_match(q, q, mask, S, S))
    self.assertFalse(attention_op.static_causal_local_shapes_match(q[:, :, : S // 2], q, mask, S, S))
    self.assertFalse(attention_op.static_causal_local_shapes_match(q, q[:, :, : S // 2], mask, S, S))
    self.assertFalse(attention_op.static_causal_local_shapes_match(q, q, mask[:, : S // 2], S, S))
    self.assertFalse(attention_op.static_causal_local_shapes_match(q, q, mask[:, :, : S // 2], S, S))

  def test_sharded_shapes_inside_shard_map_fall_back(self):
    with mock.patch.object(attention_op, "static_causal_local_shapes_match", return_value=False) as check:
      _, counts = self._dynamic_forward_counts(_make_op(True))
    check.assert_called()
    self.assertEqual(counts["pallas_in_loop"], 1)

  def test_config_requires_the_indexer(self):
    with self.assertRaisesRegex(ValueError, "splash_static_causal_partial_mask"):
      _make_op(True, use_indexer=False)


class StaticCausalPartialMaskConfigTest(unittest.TestCase):
  """MaxTextConfig.validate_splash_static_causal_partial_mask branch by branch."""

  _VALID = {
      "splash_static_causal_partial_mask": True,
      "use_indexer": True,
      "attention": "flash",
      "use_tokamax_splash": True,
      "attention_type": "mla",
      "use_multimodal": False,
      "max_target_length": 4096,
      "sa_block_q": 2048,
      "sa_block_kv": 2048,
      "sa_block_q_dkv": 2048,
      "sa_block_kv_dkv": 8192,  # clamped to max_target_length, like create_sa_config
  }

  def _validate(self, **overrides):
    config_types.MaxTextConfig.validate_splash_static_causal_partial_mask(
        types.SimpleNamespace(**{**self._VALID, **overrides})
    )

  def test_accepts(self):
    self._validate()
    self._validate(attention="autoselected")
    self._validate(attention_type="global")
    self._validate(splash_static_causal_partial_mask=False, use_indexer=False, attention="dot_product")

  def test_rejects(self):
    for overrides, message in (
        ({"use_indexer": False}, "use_indexer"),
        ({"attention": "dot_product"}, "attention='flash'"),
        ({"use_tokamax_splash": False}, "use_tokamax_splash"),
        ({"attention_type": "compressed"}, "attention_type"),
        ({"attention_type": "local_sliding"}, "attention_type"),
        ({"attention_type": "chunk"}, "attention_type"),
        ({"use_multimodal": True}, "use_multimodal"),
        ({"sa_block_kv": 1536}, "sa_block_kv=1536"),
        ({"sa_block_q_dkv": 3000}, "sa_block_q_dkv=3000"),
    ):
      with self.subTest(**overrides):
        with self.assertRaisesRegex(ValueError, message):
          self._validate(**overrides)

  def test_rejects_tokamax_without_the_private_helpers(self):
    with mock.patch.dict(tokamax_mask_info.__dict__):
      del tokamax_mask_info.__dict__["_process_dynamic_mask"]
      with self.assertRaisesRegex(ValueError, "_process_dynamic_mask"):
        self._validate()


if __name__ == "__main__":
  unittest.main()
