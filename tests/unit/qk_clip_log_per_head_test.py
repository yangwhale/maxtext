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

"""qk_clip_log_per_head: per-(layer,) head max logit and QK-Clip scale, read the same way the clip reads them."""

import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from maxtext.utils import qk_clip_utils


class PerHeadStatsTest(unittest.TestCase):

  def setUp(self):
    rng = np.random.default_rng(0)
    self.L, self.B, self.H, self.R, self.D = 3, 2, 4, 8, 6
    self.ml_scan = rng.uniform(0.05, 0.2, (self.L, self.B, self.H)).astype(np.float32)
    self.ml_one = rng.uniform(0.05, 0.2, (self.B, self.H)).astype(np.float32)
    self.params = {
        "decoder": {
            "moe_layers": {"self_attention": {"wq_b": {"kernel": jnp.zeros((self.R, self.L, self.H, self.D))},
                                               "wkv_b": {"kernel": jnp.zeros((self.R, self.L, self.H, self.D))}}},
            "mtp": {"self_attention": {"wq_b": {"kernel": jnp.zeros((self.R, self.H, self.D))}}},
            "other": {"kernel": jnp.zeros((2, 2))},
        }
    }
    self.inter = {
        "decoder": {
            "moe_layers": {"self_attention": {"attention_op": {"max_logits": jnp.asarray(self.ml_scan)}}},
            "mtp": {"self_attention": {"max_logits": (jnp.asarray(self.ml_one),)}},
        }
    }

  def _cfg(self, mcore):
    return SimpleNamespace(qk_clip_threshold=0.1, qk_clip_mcore_denominator=mcore, param_scan_axis=1)

  def test_matches_reference(self):
    for mcore, eps in ((True, 0.0), (False, 1e-6)):
      out = qk_clip_utils.per_head_stats(self.params, self.inter, self._cfg(mcore))
      self.assertEqual(sorted(out), ["decoder/moe_layers/self_attention", "decoder/mtp/self_attention"])
      s = self.ml_scan.astype(np.float64).max(axis=1)
      got = np.asarray(out["decoder/moe_layers/self_attention"])
      np.testing.assert_allclose(got[0], s, rtol=0)
      np.testing.assert_allclose(got[1], np.minimum(1.0, 0.1 / (s + eps)), rtol=1e-6)
      s1 = self.ml_one.astype(np.float64).max(axis=0)
      got1 = np.asarray(out["decoder/mtp/self_attention"])
      np.testing.assert_allclose(got1[0], s1, rtol=0)
      np.testing.assert_allclose(got1[1], np.minimum(1.0, 0.1 / (s1 + eps)), rtol=1e-6)

  def test_same_scale_as_the_clip(self):
    cfg = self._cfg(True)
    out = qk_clip_utils.per_head_stats(self.params, self.inter, cfg)
    w = self.params["decoder"]["moe_layers"]["self_attention"]["wq_b"]["kernel"]
    k = qk_clip_utils._kernel_scale(jnp.asarray(self.ml_scan), w, 0.1, 1, 0.0)  # pylint: disable=protected-access
    got = np.asarray(out["decoder/moe_layers/self_attention"][1])
    np.testing.assert_array_equal(np.asarray(k).reshape(self.L, self.H), got)

  def test_scanned_shape_mismatch_raises(self):
    bad = {"decoder": {"moe_layers": {"self_attention": {"attention_op": {"max_logits": jnp.zeros((2, self.B, self.H))}}}}}
    with self.assertRaises(ValueError):
      qk_clip_utils.per_head_stats(self.params, bad, self._cfg(True))


if __name__ == "__main__":
  unittest.main()
