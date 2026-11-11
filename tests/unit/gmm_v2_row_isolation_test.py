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

"""TPU test: the gmm_v2 kernels' VJP reads no row outside the groups into a valid result.

moe_router_grad_from_wo_input skips the [rows, hidden] zeroing of the fanned-out cotangent on the gmm_v2 path
(mask_fan_out_rows=False) because of this property: the SparseCore gather leaves the rows outside the shard's range
unwritten, so they may hold NaN. Checked on the real kernels (they need a TPU; skipped elsewhere): with NaN in the
cotangent rows outside the groups (including rows that share a tile with a group's last rows),
  - the weight gradient (tgmm) is finite and bit-identical to the run with finite rows there, and
  - the input gradient's rows inside the groups are bit-identical (the rows outside are not checked: they are
    masked downstream).
Positive control: NaN in a cotangent row inside a group does reach the weight gradient.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.kernels.megablox import ops as mblx

_G, _M, _K, _N = 4, 1024, 256, 512
# 612 rows in groups; the last group ends at row 612, inside the 128-row tile [512, 640), so rows 612-639 are
# outside the groups but share a tile with group 2; 412 rows outside in all; one empty group.
_GROUP_SIZES = (200, 312, 100, 0)
# (fwd, dlhs, drhs) x (m, k, n), as moe.py passes it.
_TILING = (128, 128, 128) * 3


def _vjp(g, group_offset=0):
  k = jax.random.split(jax.random.PRNGKey(0), 2)
  lhs = jax.random.normal(k[0], (_M, _K), jnp.float32).astype(jnp.bfloat16)
  rhs = (jax.random.normal(k[1], (_G, _K, _N)) * 0.1).astype(jnp.bfloat16)
  gs = jnp.array(_GROUP_SIZES, jnp.int32)

  def f(a, b):
    return mblx.gmm(
        a,
        b,
        gs,
        preferred_element_type=jnp.bfloat16,
        tiling=_TILING,
        group_offset=group_offset,
        use_tokamax_backend=True,
        use_gmm_v2=True,
    )

  _, vjp = jax.vjp(f, lhs, rhs)
  dlhs, drhs = vjp(g)
  return np.asarray(dlhs.astype(jnp.float32)), np.asarray(drhs.astype(jnp.float32))


def _cotangent():
  return jax.random.normal(jax.random.PRNGKey(1), (_M, _N), jnp.float32).astype(jnp.bfloat16)


@unittest.skipIf(jax.default_backend() != "tpu", "the gmm_v2 kernels need a TPU")
class GmmV2RowIsolationTest(unittest.TestCase):
  """NaN cotangent rows outside the groups do not reach a valid gradient."""

  def test_rows_outside_groups(self):
    g = _cotangent()
    inside = np.arange(_M) < sum(_GROUP_SIZES)
    g_nan = jnp.where(jnp.asarray(inside)[:, None], g, jnp.nan).astype(jnp.bfloat16)
    dlhs, drhs = _vjp(g)
    dlhs_nan, drhs_nan = _vjp(g_nan)
    self.assertTrue(np.isfinite(drhs).all() and np.abs(drhs).sum() > 0)
    self.assertTrue(np.isfinite(drhs_nan).all(), "NaN rows outside the groups reached the weight gradient")
    np.testing.assert_array_equal(drhs_nan, drhs)
    np.testing.assert_array_equal(dlhs_nan[inside], dlhs[inside])

  def test_nan_inside_a_group_reaches_the_weight_gradient(self):
    # Positive control: the injection is visible to the kernel when the row is inside a group.
    g = _cotangent()
    g_nan = g.at[5].set(jnp.nan)
    _, drhs_nan = _vjp(g_nan)
    self.assertFalse(np.isfinite(drhs_nan).all(), "a NaN row inside group 0 did not reach the weight gradient")


if __name__ == "__main__":
  unittest.main()
