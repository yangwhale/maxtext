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

"""qk_clip_max_logit_exclude_sink: the QK-Clip max logit recomputed over attended positions, sink left out."""

import unittest

import jax.numpy as jnp
import numpy as np

from maxtext.layers.attention_op import max_logits_excluding_sink


def _reference(q, k, allowed):
  logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64))
  return np.where(allowed[:, None], logits, -np.inf).max(axis=(2, 3))


class MaxLogitsExcludingSinkTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    self.b, self.s, self.h, self.d = 2, 16, 3, 8
    self.q = rng.standard_normal((self.b, self.s, self.h, self.d)).astype(np.float32) * 0.1
    self.k = rng.standard_normal((self.b, self.s, self.h, self.d)).astype(np.float32) * 0.1
    self.causal = np.tril(np.ones((self.s, self.s), bool))[None].repeat(self.b, 0)

  def test_causal_matches_reference(self):
    got = np.asarray(max_logits_excluding_sink(jnp.asarray(self.q), jnp.asarray(self.k)))
    np.testing.assert_allclose(got, _reference(self.q, self.k, self.causal), rtol=1e-5, atol=1e-6)

  def test_indexer_mask_and_segments(self):
    rng = np.random.default_rng(1)
    keep = rng.random((self.b, self.s, self.s)) < 0.5
    keep[:, np.arange(self.s), np.arange(self.s)] = True  # every row attends at least itself
    additive = np.where(keep, 0.0, -1e30).astype(np.float32)
    seg = np.ones((self.b, self.s), np.int32)
    seg[:, self.s // 2 :] = 2
    allowed = self.causal & keep & (seg[:, :, None] == seg[:, None, :])
    got = np.asarray(
        max_logits_excluding_sink(jnp.asarray(self.q), jnp.asarray(self.k), jnp.asarray(additive), jnp.asarray(seg))
    )
    np.testing.assert_allclose(got, _reference(self.q, self.k, allowed), rtol=1e-5, atol=1e-6)

  def test_sink_not_included(self):
    """All logits negative: a running max started from a sink of 0 would report 0; this reports the true max."""
    q = -np.abs(self.q)
    k = np.abs(self.k)
    got = np.asarray(max_logits_excluding_sink(jnp.asarray(q), jnp.asarray(k)))
    self.assertTrue((got < 0).all())
    np.testing.assert_allclose(got, _reference(q, k, self.causal), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
