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

"""Switches that make MaxText's optimizer / loss details match stock Megatron-Core.

Each switch: off = unchanged, on = Megatron-Core's formula on the same tensors.
- muon_mcore_nesterov, muon_mcore_ns_normalize, muon_adam_plain: against updates produced by Megatron-Core's
  TensorParallelMuon (emerging_optimizers 0.3.0) and torch Adam on the same gradients
  (tests/assets/muon_mcore/, generator gen_ref.py next to it).
- qk_clip_mcore_denominator: min(1, tau / max_logit).
- reduce_microbatch_aux: QK-Clip max_logits are a max over gradient-accumulation microbatches (bug fix).
- indexer_kd_mcore_log: Megatron-Core's indexer KL form, against a float64 reference of that formula.
- indexer_kd_generalized_kl: the recorded indexer KL as sum(p (expm1(r) - r)), r = log q - log p; same gradients.
"""

import os
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from optax.contrib import MuonDimensionNumbers as mdn
from optax.contrib._muon import muon as optax_muon

from maxtext.layers.attention_mla import MLA
from maxtext.optimizers.muon_mcore_compat import muon_compat
from maxtext.utils import qk_clip_utils
from maxtext.utils.gradient_accumulation import reduce_microbatch_aux

REF = os.path.join(os.path.dirname(__file__), "..", "assets", "muon_mcore", "muon_mcore_reference.npz")
MATS = ("wide", "tall", "square")
KW = {
    "learning_rate": 1.0,
    "beta": 0.95,
    "eps": 1e-10,
    "adam_b1": 0.9,
    "adam_b2": 0.95,
    "ns_coeffs": (3.4445, -4.7750, 2.0315),
}


def _run(tx, ref):
  """Feed the reference gradients (torch [out, in] -> MaxText [in, out]) for T steps; return updates per step."""
  params = {k: jnp.zeros(ref[f"{k}_grads"].shape[:0:-1]) for k in MATS}
  params["scalar"] = jnp.zeros(ref["scalar_grads"].shape[1:])
  state = tx.init(params)
  outs = []
  for t in range(ref["scalar_grads"].shape[0]):
    g = {k: jnp.asarray(ref[f"{k}_grads"][t].T) for k in MATS}
    g["scalar"] = jnp.asarray(ref["scalar_grads"][t])
    up, state = tx.update(g, state, params)
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, up)
    outs.append(jax.tree_util.tree_map(np.asarray, up))
  return outs


def _rel(a, b):
  return float(np.linalg.norm(a - b) / np.linalg.norm(b))


DN = {k: mdn((0,), (1,)) for k in MATS} | {"scalar": None}


class MuonMcoreTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.ref = dict(np.load(REF))

  def test_switches_on_match_megatron_core(self):
    tx = muon_compat(
        **KW, muon_weight_dimension_numbers=DN, mcore_nesterov=True, adam_nesterov=False, mcore_ns_normalize=True
    )
    outs = _run(tx, self.ref)
    for t, up in enumerate(outs):
      for k in MATS:
        self.assertLess(_rel(up[k].T, self.ref[f"{k}_updates"][t]), 2e-5, (k, t))
      self.assertLess(_rel(up["scalar"], self.ref["scalar_updates"][t]), 1e-5, t)

  def test_switches_off_equal_optax_and_differ_from_megatron_core(self):
    off = _run(muon_compat(**KW, muon_weight_dimension_numbers=DN), self.ref)
    opt = _run(optax_muon(**KW, muon_weight_dimension_numbers=DN), self.ref)
    for a, b in zip(off, opt):
      for k in a:
        np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-9)
    # Negative control: the default (optax) form is far from Megatron-Core after step 1.
    self.assertGreater(_rel(off[1]["wide"].T, self.ref["wide_updates"][1]), 1e-2)
    self.assertGreater(_rel(off[0]["scalar"], self.ref["scalar_updates"][0]), 1e-1)  # NAdam vs Adam

  def test_each_switch_alone(self):
    nest = _run(muon_compat(**KW, muon_weight_dimension_numbers=DN, mcore_nesterov=True), self.ref)
    for t in range(len(nest)):
      self.assertLess(_rel(nest[t]["wide"].T, self.ref["wide_updates"][t]), 2e-5, t)  # normalization alone is ~1e-6
    plain = _run(muon_compat(**KW, muon_weight_dimension_numbers=DN, adam_nesterov=False), self.ref)
    for t in range(len(plain)):
      self.assertLess(_rel(plain[t]["scalar"], self.ref["scalar_updates"][t]), 1e-5, t)


class QKClipDenominatorTest(unittest.TestCase):

  def test_denominator(self):
    s = jnp.array([50.0, 100.0, 400.0, 1e-5])
    np.testing.assert_allclose(
        qk_clip_utils._clip_scale(s, 100.0, 0.0), [1.0, 1.0, 0.25, 1.0]  # pylint: disable=protected-access
    )
    # The default +1e-6 only shows when the max logit is tiny: tau 1e-6, max logit 1e-5 -> 0.1 vs 1/11.
    np.testing.assert_allclose(qk_clip_utils._clip_scale(s[3:], 1e-6, 0.0), [0.1], rtol=1e-6)  # pylint: disable=protected-access
    np.testing.assert_allclose(
        qk_clip_utils._clip_scale(s[3:], 1e-6, 1e-6), [1 / 11], rtol=1e-6  # pylint: disable=protected-access
    )


class MicrobatchMaxLogitsTest(unittest.TestCase):

  def test_max_logits_are_maxed_others_summed(self):
    scanned = {
        "xent_sum": jnp.array([1.0, 2.0]),
        "intermediate_outputs": {
            "decoder": {
                "layers": {"self_attention": {"attention_op": {"max_logits": jnp.array([[[3.0, 9.0]], [[7.0, 2.0]]])}}}
            },
            "moe": {"counts": jnp.array([[1, 2], [3, 4]])},
        },
    }
    out = reduce_microbatch_aux(scanned)
    self.assertEqual(float(out["xent_sum"]), 3.0)
    np.testing.assert_array_equal(
        out["intermediate_outputs"]["decoder"]["layers"]["self_attention"]["attention_op"]["max_logits"], [[7.0, 9.0]]
    )
    np.testing.assert_array_equal(out["intermediate_outputs"]["moe"]["counts"], [4, 6])


def _mcore_kd_reference(index_score, attn_scores_masked):
  """float64 Megatron-Core form (dsa_indexer_loss): p = sum_h softmax / max(sum, 1e-10);
  KL = sum p (log max(p, 1e-10) - log_softmax(index)) over p > 0; mean over rows."""
  a = attn_scores_masked.astype(np.float64)
  a = np.exp(a - a.max(-1, keepdims=True))
  a = a / a.sum(-1, keepdims=True)
  p = a.sum(1)
  p = p / np.maximum(p.sum(-1, keepdims=True), 1e-10)
  x = index_score.astype(np.float64)
  m = x.max(-1, keepdims=True)
  logq = x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
  kl = np.where(p > 0, p * (np.log(np.maximum(p, 1e-10)) - np.where(p > 0, logq, 0.0)), 0.0)
  return float(kl.sum(-1).mean())


class IndexerKDTest(unittest.TestCase):

  def _inputs(self):
    rng = np.random.default_rng(0)
    b, t, h, d = 2, 8, 3, 4
    q = rng.standard_normal((b, t, h, d)).astype(np.float32)
    k = rng.standard_normal((b, t, h, d)).astype(np.float32)
    score = (30.0 * rng.standard_normal((b, t, t))).astype(np.float32)  # peaked indexer: q tiny where p is not
    causal = np.where(np.tril(np.ones((t, t), bool)), 0.0, -1e30).astype(np.float32)
    mask = np.broadcast_to(causal, (b, t, t)).copy()
    return q, k, score, mask

  def _loss(self, mcore, q, k, score, mask):
    fake = SimpleNamespace(
        config=SimpleNamespace(mla_qk_head_chunk_size=0, matmul_precision="highest", indexer_kd_mcore_log=mcore)
    )
    return float(
        MLA.calculate_indexer_loss(
            fake, jnp.asarray(score), jnp.asarray(q), jnp.asarray(k), None, jnp.asarray(mask), True, 1.0
        )
    )

  def test_mcore_form(self):
    q, k, score, mask = self._inputs()
    attn = np.einsum("bthd,bshd->bhts", q, k) + mask[:, None]
    ref = _mcore_kd_reference(score + mask, attn)
    got = self._loss(True, q, k, score, mask)
    self.assertAlmostEqual(got, ref, delta=1e-5 * max(1.0, abs(ref)))
    # Off: the default writing, log(p + 1e-8) - log(q + 1e-8); not the same number on these inputs.
    self.assertGreater(abs(self._loss(False, q, k, score, mask) - got), 1e-4)


if __name__ == "__main__":
  unittest.main()


class IndexerKDWeightTest(unittest.TestCase):
  """indexer_kd_mcore_weight: Megatron-Core sum_rows KL * c / N_global vs MaxText mean_rows * N_mb / N_global."""

  def test_term(self):
    from maxtext.trainers.pre_train.train import indexer_kd_mcore_term  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    kl_rows = rng.random(16)  # per-row KL already times the coefficient, 2 rows x 8 positions flattened
    valid = 13  # e.g. 3 eod positions masked out of 16
    mean_rows = float(kl_rows.mean())
    # Manual GA: the objective is divided by N_global later; Megatron-Core contributes sum_rows KL.
    np.testing.assert_allclose(float(indexer_kd_mcore_term(mean_rows, 16, valid, True)), kl_rows.sum(), rtol=1e-6)
    # Without GA the loss is per token of this microbatch: sum_rows KL / valid.
    np.testing.assert_allclose(
        float(indexer_kd_mcore_term(mean_rows, 16, valid, False)), kl_rows.sum() / valid, rtol=1e-6
    )
    # The default weighting (by valid tokens) differs by 16 / 13 here.
    self.assertGreater(kl_rows.sum() / (mean_rows * valid), 1.2)


class MTPPerRankTest(unittest.TestCase):
  """mtp_loss_mcore_per_rank: Megatron-Core rescales each rank's (= row's) MTP loss by its own original / rolled counts."""

  def _inter(self, rows_loss, rows_rolled, rows_orig):
    scaled = np.sum(rows_loss * rows_orig / np.maximum(rows_rolled, 1))
    return {
        "mtp_losses": {
            "mtp_block": {
                "losses": jnp.array([rows_loss.sum()]),
                "weights": jnp.array([rows_rolled.sum()]),
                "rank_scaled_losses": jnp.array([scaled]),
                "orig_tokens": jnp.asarray(rows_orig.sum()),
            }
        }
    }

  def test_per_rank_vs_pooled(self):
    from maxtext.layers.multi_token_prediction import calculate_mtp_loss  # pylint: disable=import-outside-toplevel

    loss = np.array([30.0, 50.0])  # MTP loss sums of two rows (devices)
    rolled = np.array([9.0, 6.0])  # rolled valid tokens differ per row (eod)
    orig = np.array([10.0, 7.0])
    inter = self._inter(loss, rolled, orig)
    on = float(calculate_mtp_loss(inter, SimpleNamespace(mtp_loss_scaling_factor=0.1, mtp_loss_mcore_per_rank=True)))
    off = float(calculate_mtp_loss(inter, SimpleNamespace(mtp_loss_scaling_factor=0.1, mtp_loss_mcore_per_rank=False)))
    # Times the microbatch's main-loss tokens (manual-GA scaling) = Megatron-Core's sum over ranks.
    np.testing.assert_allclose(on * orig.sum(), 0.1 * np.sum(loss * orig / rolled), rtol=1e-6)
    np.testing.assert_allclose(off, 0.1 * loss.sum() / rolled.sum(), rtol=1e-6)  # unchanged default
    self.assertGreater(abs(on - off) / off, 1e-3)


class IndexerFp32TeacherTest(unittest.TestCase):
  """indexer_fp32_scores: the KD teacher q.k from BF16 query / key is a true FP32 product (Megatron-Core bmm on .float())."""

  def test_teacher_fp32(self):
    rng = np.random.default_rng(1)
    b, t, h, d = 1, 16, 2, 64
    q = jnp.asarray(rng.standard_normal((b, t, h, d)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((b, t, h, d)), jnp.bfloat16)
    score = jnp.asarray(rng.standard_normal((b, t, t)), jnp.float32)
    mask = jnp.asarray(np.broadcast_to(np.where(np.tril(np.ones((t, t), bool)), 0.0, -1e30), (b, t, t)), jnp.float32)

    def loss(fp32):
      cfg = SimpleNamespace(
          mla_qk_head_chunk_size=0, matmul_precision="default", indexer_kd_mcore_log=True, indexer_fp32_scores=fp32
      )
      return float(MLA.calculate_indexer_loss(SimpleNamespace(config=cfg), score, q, k, None, mask, True, 1.0))

    qf, kf = np.asarray(q, np.float64), np.asarray(k, np.float64)
    attn = np.einsum("bthd,bshd->bhts", qf, kf) + np.asarray(mask, np.float64)[:, None]
    ref = _mcore_kd_reference(np.asarray(score + mask), attn)
    self.assertAlmostEqual(loss(True), ref, delta=2e-6 * max(1.0, abs(ref)))
    # Off: q.k in the compute dtype (BF16 output), which is not the FP32 teacher.
    self.assertGreater(abs(loss(False) - ref), 1e-5 * abs(ref))


class IndexerKDGeneralizedTest(unittest.TestCase):
  """indexer_kd_generalized_kl: only the recorded value changes; it is first-order insensitive to sum(p) != 1 and to a
  uniform shift of log q, the two FP32 biases that push the recorded KL up when the KL is ~1e-4 per row."""

  def _setup(self):
    from maxtext.common.common_types import DEFAULT_MASK_VALUE  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(0)
    b, t, h, d = 1, 512, 8, 64
    q = (0.6 * rng.standard_normal((b, t, h, d))).astype(np.float32)
    k = (0.6 * rng.standard_normal((b, t, h, d))).astype(np.float32)
    causal = np.tril(np.ones((t, t), bool))
    mask = np.broadcast_to(np.where(causal, 0.0, DEFAULT_MASK_VALUE), (b, t, t)).astype(np.float32)
    x = np.where(causal, np.einsum("bthd,bshd->bhts", q.astype(np.float64), k.astype(np.float64)), -np.inf)
    e = np.exp(x - x.max(-1, keepdims=True))
    a = (e / e.sum(-1, keepdims=True)).sum(1)
    p = a / a.sum(-1, keepdims=True)
    # an indexer close to the teacher, so the KL is small (~5e-5 per row) as in training
    score = np.where(causal, np.log(np.maximum(p, 1e-30)) + 0.01 * rng.standard_normal((b, t, t)), 0.0).astype(np.float32)
    z = np.where(causal, score.astype(np.float64), -np.inf)
    lq = z - z.max(-1, keepdims=True)
    lq = lq - np.log(np.exp(lq).sum(-1, keepdims=True))
    ok = causal & (p > 0)
    ref = float(np.where(ok, p * (np.log(np.maximum(p, 1e-10)) - np.where(ok, lq, 0.0)), 0.0).sum(-1).mean())
    return q, k, score, mask, ref

  def _loss(self, gen, q, k, score, mask):
    cfg = SimpleNamespace(
        mla_qk_head_chunk_size=0,
        matmul_precision="highest",
        indexer_kd_mcore_log=True,
        indexer_fp32_scores=True,
        transcendental_accuracy_highest=True,
        indexer_kd_generalized_kl=gen,
    )
    return MLA.calculate_indexer_loss(
        SimpleNamespace(config=cfg), score, jnp.asarray(q), jnp.asarray(k), None, jnp.asarray(mask), True, 1.0
    )

  def test_same_gradient(self):
    q, k, score, mask, _ = self._setup()
    s = jnp.asarray(score)
    g_off = jax.grad(lambda s: self._loss(False, q, k, s, mask))(s)
    g_on = jax.grad(lambda s: self._loss(True, q, k, s, mask))(s)
    scale = float(jnp.max(jnp.abs(g_off)))
    self.assertLessEqual(float(jnp.max(jnp.abs(g_on - g_off))), 1e-7 * scale)

  def test_value(self):
    q, k, score, mask, ref = self._setup()
    on = float(self._loss(True, q, k, jnp.asarray(score), mask))
    off = float(self._loss(False, q, k, jnp.asarray(score), mask))
    # On CPU the KL form is already ~2e-4 high here (sum(p) - 1 ~ 1e-8 in FP32 against a KL of ~5e-5); the
    # generalized form is limited by the FP32 inputs themselves (an FP64 evaluation of either form on the same FP32
    # p and log q is within 1e-6 of the reference).
    self.assertLess(abs(on / ref - 1), 2e-6)
    self.assertGreater(abs(off / ref - 1), 1e-4)

  def test_invariance(self):
    """The per-element form, fed p scaled by (1 + delta) and log q shifted by eps, moves by O(delta KL), not delta."""
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(64) * 50.0)
    logq = np.log(p) + 0.01 * rng.standard_normal(64)
    logq -= np.log(np.exp(logq).sum())

    def gen(pp, lq):
      r = lq - np.log(pp)
      return float(np.sum(pp * (np.expm1(r) - r)))

    def kl(pp, lq):
      return float(np.sum(pp * (np.log(pp) - lq)))

    base = kl(p, logq)
    delta, eps = 2e-8, -2e-8
    self.assertLess(abs(gen(p * (1 + delta), logq + eps) / base - 1), 1e-6)
    self.assertGreater(abs(kl(p * (1 + delta), logq + eps) / base - 1), 1e-4)
