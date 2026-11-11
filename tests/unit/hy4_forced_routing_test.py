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
"""debug_forced_routing_deepseek: router replay on the DeepSeek decoder (IndexCache blocks, remainder block, MTP).

On hy4-tiny (1 dense layer, one scanned IndexCache block of 4 MoE layers, a remainder block of 2, one MTP layer):
- replaying the model's own top-k (recorded with debug_routing_digest) reproduces the unforced logits and MTP loss
  bit for bit: the weights come from the pre-bias scores and are normalized exactly as DeepSeek routing does;
- every routed MoE sows the ids it used, equal to its slice of the input, and agrees with its own top-k on all tokens;
- changing one layer's replayed ids changes the output and only that layer's agreement; changing the MTP slot changes
  the MTP loss but not the main logits;
- off (default), forced routing on a DeepSeek decoder still raises.
"""

import re
import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.common.common_types import DecoderBlockType
from maxtext.configs import pyconfig
from maxtext.configs.types import check_forced_routing_support
from maxtext.utils import maxtext_utils, model_creation_utils, routing_debug
from tests.utils.test_helpers import get_test_config_path

S = 32


def _config(**kw):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="hy4_forced_routing_test",
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      model_name="hy4-tiny",
      per_device_batch_size=1,
      max_target_length=S,
      max_prefill_predict_length=S,
      dtype="bfloat16",
      weight_dtype="float32",
      attention="dot_product",
      sparse_matmul=False,
      megablox=False,
      **kw,
  )


def _run(cfg, forced=None, fit=None):
  """Fresh model (same init every time), one forward; returns (logits, flat Intermediates)."""
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  rng = np.random.default_rng(0)
  tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, (1, S)), jnp.int32)
  pos = jnp.arange(S, dtype=jnp.int32)[None, :]
  seg = jnp.ones_like(tokens)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))
    kw = {} if forced is None else {"forced_routed_experts": forced}
    if fit is not None:
      kw["forced_indexer_topk"] = fit
    out = model(
        tokens,
        pos,
        decoder_segment_ids=seg,
        enable_dropout=False,
        model_mode="train",
        decoder_target_tokens=tokens,
        decoder_target_mask=seg,
        **kw,
    )
  logits = out[0] if isinstance(out, tuple) else out
  inter = {"/".join(map(str, p)): v.get_value() for p, v in nnx.to_flat_state(nnx.state(model, nnx.Intermediate))}
  return np.asarray(logits, np.float32), inter


def _grads(cfg, forced=None, fit=None):
  """Loss (main + nothing else) and parameter gradients of one forward / backward."""
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  rng = np.random.default_rng(0)
  tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, (1, S)), jnp.int32)
  pos = jnp.arange(S, dtype=jnp.int32)[None, :]
  seg = jnp.ones_like(tokens)
  kw = {} if forced is None else {"forced_routed_experts": forced}
  if fit is not None:
    kw["forced_indexer_topk"] = fit
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    model = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))

    def loss_fn(m):
      out = m(
          tokens,
          pos,
          decoder_segment_ids=seg,
          enable_dropout=False,
          model_mode="train",
          decoder_target_tokens=tokens,
          decoder_target_mask=seg,
          **kw,
      )
      logits = out[0] if isinstance(out, tuple) else out
      logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
      return -jnp.mean(jnp.take_along_axis(logp, tokens[..., None], axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
  return float(loss), {"/".join(map(str, p)): np.asarray(v.get_value()) for p, v in nnx.to_flat_state(grads)}


def _pick(inter, name):
  return {k: v for k, v in inter.items() if name in k}


def _mtp_loss(inter):
  vals = [np.asarray(v, np.float32) for k, v in inter.items() if k.endswith("mtp_block/losses")]
  assert vals, sorted(inter)
  return np.concatenate([v.ravel() for v in vals])


def _layer_of(path):
  """HF MoE layer (1..6) or 7 (MTP) of a sown forced_routing_used path (hy4-tiny layout)."""
  if "mtp" in path:
    return 7
  sub = int(re.search(r"/layers_(\d+)/", path).group(1))
  return 1 + sub if "moe_layers_remainder" not in path else 5 + sub


class Hy4ForcedRoutingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    routing_debug.LOG.clear()
    cls.logits0, cls.inter0 = _run(_config(debug_routing_digest=True))
    log = [np.asarray(x) for x in routing_debug.LOG]
    routing_debug.LOG.clear()
    # forward order: IndexCache block (layers 1-4), remainder (5-6), MTP
    assert len(log) == 7, [x.shape for x in log]
    cfg = _config(debug_forced_routing_deepseek=True)
    cls.cfg = cfg
    k = cfg.num_experts_per_tok
    fre = np.zeros((1, S, cfg.num_decoder_layers + cfg.mtp_num_layers, k), np.int32)  # slot 0 (dense) unused
    for i, ids in enumerate(log):
      fre[:, :, 1 + i] = ids.reshape(1, S, k)
    cls.fre = fre
    cls.logits1, cls.inter1 = _run(cfg, jnp.asarray(fre))

  def test_replaying_own_routing_is_bit_identical(self):
    np.testing.assert_array_equal(self.logits1, self.logits0)
    np.testing.assert_array_equal(_mtp_loss(self.inter1), _mtp_loss(self.inter0))

  def test_gradients_bit_identical_under_remat(self):
    # remat_policy custom (the T1 setting): the IndexCache sub-layers rematerialize with the replayed ids passed in
    loss0, g0 = _grads(_config(remat_policy="custom"))
    loss1, g1 = _grads(_config(remat_policy="custom", debug_forced_routing_deepseek=True), jnp.asarray(self.fre))
    self.assertEqual(loss0, loss1)
    self.assertEqual(sorted(g0), sorted(g1))
    for k, v in g0.items():
      np.testing.assert_array_equal(g1[k], v, err_msg=k)

  def test_audit_sow(self):
    used = _pick(self.inter1, "forced_routing_used")
    agree = _pick(self.inter1, "forced_routing_natural_agree")
    seen = set()
    for path, v in used.items():
      v = np.asarray(v)
      v = v.reshape((-1,) + v.shape[-3:])[0]  # drop the scan / sow-tuple axes (one call per layer)
      layer = _layer_of(path)
      seen.add(layer)
      np.testing.assert_array_equal(v, self.fre[:, :, layer], err_msg=path)
    self.assertEqual(seen, set(range(1, 8)))
    self.assertEqual(len(agree), 7)
    for path, v in agree.items():
      self.assertEqual(float(np.ravel(np.asarray(v))[0]), 1.0, path)
    self.assertFalse(_pick(self.inter0, "forced_routing_used"))  # off: nothing sown

  def test_one_layer_changed(self):
    fre = self.fre.copy()
    fre[:, :, 5] = (fre[:, :, 5] + 1) % self.cfg.num_experts  # remainder block, first layer
    logits, inter = _run(self.cfg, jnp.asarray(fre))
    self.assertFalse(np.array_equal(logits, self.logits0))
    for path, v in _pick(inter, "forced_routing_natural_agree").items():
      a = float(np.ravel(np.asarray(v))[0])
      if _layer_of(path) == 5:
        self.assertLess(a, 1.0, path)
      elif _layer_of(path) < 5:
        self.assertEqual(a, 1.0, path)  # upstream layers see the same input

  def test_mtp_slot_changed(self):
    fre = self.fre.copy()
    fre[:, :, 7] = (fre[:, :, 7] + 1) % self.cfg.num_experts
    logits, inter = _run(self.cfg, jnp.asarray(fre))
    np.testing.assert_array_equal(logits, self.logits0)
    self.assertFalse(np.array_equal(_mtp_loss(inter), _mtp_loss(self.inter0)))

  def test_off_still_raises(self):
    with self.assertRaises(NotImplementedError):
      check_forced_routing_support(DecoderBlockType.DEEPSEEK)
    check_forced_routing_support(DecoderBlockType.DEEPSEEK, allow_deepseek=True)
    with self.assertRaises(NotImplementedError):
      _run(_config(), jnp.asarray(self.fre))

  def test_wrong_layer_axis_raises(self):
    with self.assertRaises(ValueError):
      _run(self.cfg, jnp.asarray(self.fre[:, :, :7]))


def _indexer_slot(path):
  """Layer slot of an indexer layer's sow path (hy4-tiny: dense layer 0, MoE blocks start at 1 and 5, MTP 7)."""
  if "mtp" in path:
    return 7
  if "dense_layers" in path:
    return 0
  return 5 if "moe_layers_remainder" in path else 1


class Hy4ForcedIndexerTest(unittest.TestCase):
  """debug_forced_indexer_topk on hy4-tiny (indexer_topk 16 < sequence 32, so the indexer selects)."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.logits0, cls.inter0 = _run(_config(debug_sow_indexer=True))
    cfg = _config(debug_forced_indexer_topk=True)
    cls.cfg = cfg
    k = cfg.indexer_topk
    fit = np.full((1, S, cfg.num_decoder_layers + cfg.mtp_num_layers, k), -1, np.int32)
    masks = _pick(cls.inter0, "debug_kd_indexer_mask")
    seen = set()
    for path, m in masks.items():
      m = np.asarray(m, np.float32)
      m = m.reshape((-1,) + m.shape[-2:])[0]  # [t, s] of row 0 (one call per indexer layer)
      slot = _indexer_slot(path)
      seen.add(slot)
      for q in range(S):
        pos = np.nonzero(m[q] > -1e9)[0]
        assert len(pos) <= k
        fit[0, q, slot, : len(pos)] = pos[::-1]  # order does not matter; reversed on purpose
    assert seen == {0, 1, 5, 7}, (seen, sorted(masks))
    cls.fit = fit
    cls.logits1, cls.inter1 = _run(cfg, fit=jnp.asarray(fit))

  def test_replaying_own_selection_is_bit_identical(self):
    np.testing.assert_array_equal(self.logits1, self.logits0)
    np.testing.assert_array_equal(_mtp_loss(self.inter1), _mtp_loss(self.inter0))

  def test_gradients_bit_identical_under_remat(self):
    loss0, g0 = _grads(_config(remat_policy="custom"))
    loss1, g1 = _grads(_config(remat_policy="custom", debug_forced_indexer_topk=True), fit=jnp.asarray(self.fit))
    self.assertEqual(loss0, loss1)
    for k, v in g0.items():
      np.testing.assert_array_equal(g1[k], v, err_msg=k)

  def test_audit_sow(self):
    sums = _pick(self.inter1, "forced_indexer_index_sum")
    self.assertEqual({_indexer_slot(p) for p in sums}, {0, 1, 5, 7})
    for path, v in sums.items():
      v = np.asarray(v)
      v = v.reshape((-1,) + v.shape[-2:])[0]
      f = self.fit[:, :, _indexer_slot(path)]
      np.testing.assert_array_equal(v, np.where(f >= 0, f, 0).sum(-1), err_msg=path)
    for path, v in _pick(self.inter1, "forced_indexer_natural_agree").items():
      self.assertEqual(float(np.ravel(np.asarray(v))[0]), 1.0, path)

  def test_one_indexer_changed(self):
    fit = self.fit.copy()
    q = S - 1  # last query: drop its first selected key, take an unselected one
    row = fit[0, q, 5]
    unselected = sorted(set(range(S)) - set(row[row >= 0].tolist()))
    row[0] = unselected[0]
    logits, inter = _run(self.cfg, fit=jnp.asarray(fit))
    self.assertFalse(np.array_equal(logits, self.logits0))
    for path, v in _pick(inter, "forced_indexer_natural_agree").items():
      a = float(np.ravel(np.asarray(v))[0])
      if _indexer_slot(path) == 5:
        self.assertLess(a, 1.0, path)
      elif _indexer_slot(path) in (0, 1):
        self.assertEqual(a, 1.0, path)

  def test_off_raises(self):
    with self.assertRaises(NotImplementedError):
      _run(_config(), fit=jnp.asarray(self.fit))


class Hy4CaptureChoicesTest(unittest.TestCase):
  """debug_capture_choices: sows the layer's own choices; loss and gradients unchanged."""

  def test_gradients_unchanged(self):
    loss0, g0 = _grads(_config(remat_policy="custom"))
    loss1, g1 = _grads(_config(remat_policy="custom", debug_capture_choices=True))
    self.assertEqual(loss0, loss1)
    for k, v in g0.items():
      np.testing.assert_array_equal(g1[k], v, err_msg=k)

  def test_captured_equal_own_choices(self):
    routing_debug.LOG.clear()
    logits0, inter0 = _run(_config(debug_routing_digest=True, debug_sow_indexer=True))
    log = [np.asarray(x).reshape(S, -1) for x in routing_debug.LOG]
    routing_debug.LOG.clear()
    logits1, inter1 = _run(_config(debug_capture_choices=True))
    np.testing.assert_array_equal(logits1, logits0)
    got = {}
    for path, v in _pick(inter1, "captured_routed_experts").items():
      got[_layer_of(path)] = np.asarray(v).reshape(-1, S, log[0].shape[-1])[0]
    self.assertEqual(sorted(got), list(range(1, 8)))
    for i, ids in enumerate(log):  # forward order: layers 1-6, MTP
      np.testing.assert_array_equal(np.sort(got[1 + i], -1), np.sort(ids, -1))
    self.assertEqual(len(_pick(inter1, "captured_router_scores")), 7)
    masks = {
        _indexer_slot(p): np.asarray(m, np.float32).reshape((-1, S, S))[0]
        for p, m in _pick(inter0, "debug_kd_indexer_mask").items()
    }
    caps = {}
    for p, v in _pick(inter1, "captured_indexer_topk").items():
      v = np.asarray(v)
      caps[_indexer_slot(p)] = v.reshape((-1, S, v.shape[-1]))[0]
    self.assertEqual(sorted(caps), [0, 1, 5, 7])
    for slot, c in caps.items():
      for q in range(S):
        own = set(np.nonzero(masks[slot][q] > -1e9)[0].tolist())
        self.assertEqual(set(c[q][c[q] >= 0].tolist()), own, (slot, q))


if __name__ == "__main__":
  unittest.main()
