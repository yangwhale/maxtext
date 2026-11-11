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

"""logits_matmul_precision: matmul precision of the output head only (hy4-tiny, untied logits_dense).

On TPU an fp32 dot at precision DEFAULT runs as one bf16 pass, so logits_dot_in_fp32 alone does not give an fp32
head. CPU always computes fp32 dots in fp32, so besides the numerics this checks the precision recorded on the head
dot_general in the jaxpr, which is what the TPU compiler reads.
"""

import sys
import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers.embeddings import attend_on_embedding
from maxtext.utils import maxtext_utils, model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _config(**kw):
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="logits_matmul_precision_test",
      enable_checkpointing=False,
      log_config=False,
      skip_jax_distributed_system=True,
      model_name="hy4-tiny",
      per_device_batch_size=1,
      max_target_length=16,
      max_prefill_predict_length=16,
      dtype="bfloat16",
      weight_dtype="float32",
      attention="dot_product",
      **kw,
  )


def _build(cfg):
  mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
  with nn_partitioning.axis_rules(cfg.logical_axis_rules):
    return model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(params=0, dropout=1))


def _dot_precisions(jaxpr, out_last_dim):
  """Precisions of every dot_general (recursively) whose output's last dim is out_last_dim."""
  found = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "dot_general" and eqn.outvars[0].aval.shape[-1] == out_last_dim:
      found.append(eqn.params["precision"])
    for v in eqn.params.values():
      for sub in v if isinstance(v, (list, tuple)) else (v,):
        if isinstance(sub, jax.extend.core.ClosedJaxpr):
          found += _dot_precisions(sub.jaxpr, out_last_dim)
        elif isinstance(sub, jax.extend.core.Jaxpr):
          found += _dot_precisions(sub, out_last_dim)
  return found


class LogitsMatmulPrecisionTest(unittest.TestCase):

  def _head(self, **kw):
    cfg = _config(**kw)
    model = _build(cfg)
    # A non-bf16-representable fp32 head kernel, as HF enable_lm_head_fp32 keeps it.
    kernel = np.random.default_rng(0).standard_normal((cfg.emb_dim, cfg.vocab_size)).astype(np.float32) * 0.05
    model.decoder.logits_dense.kernel.set_value(jnp.asarray(kernel))
    y = jnp.asarray(np.random.default_rng(1).standard_normal((1, cfg.max_target_length, cfg.emb_dim)), jnp.bfloat16)

    def head(h):
      with nn_partitioning.axis_rules(cfg.logical_axis_rules):
        return model.decoder.apply_output_head(model.token_embedder, h, True, "train", normalize_y=False)

    return cfg, kernel, y, head

  def test_highest_matches_fp32_reference(self):
    cfg, kernel, y, head = self._head(logits_dot_in_fp32=True, logits_matmul_precision="highest")
    self.assertEqual(cfg.matmul_precision, "default")  # every other matmul keeps its precision
    out = np.asarray(jax.jit(head)(y), np.float64)
    ref = np.asarray(y.astype(jnp.float32), np.float64) @ kernel.astype(np.float64)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())
    # A bf16 head (kernel rounded to bf16) is visibly off: the tolerance above is meaningful.
    k_bf16 = np.asarray(jnp.asarray(kernel, jnp.bfloat16).astype(jnp.float32), np.float64)
    ref_bf16 = np.asarray(y.astype(jnp.float32), np.float64) @ k_bf16
    self.assertGreater(np.abs(ref_bf16 - ref).max(), 1e3 * 1e-6 * np.abs(ref).max())
    # The head dot carries HIGHEST, which TPU lowers to a multi-pass fp32 matmul.
    precisions = _dot_precisions(jax.make_jaxpr(head)(y).jaxpr, cfg.vocab_size)
    self.assertEqual(len(precisions), 1)
    self.assertEqual(precisions[0], (lax.Precision.HIGHEST, lax.Precision.HIGHEST))

  def test_empty_follows_matmul_precision(self):
    _, _, y, head_default = self._head(logits_dot_in_fp32=True)
    cfg, _, _, head_explicit = self._head(logits_dot_in_fp32=True, logits_matmul_precision="")
    # "" is the default and behaves as before: the head dot runs at matmul_precision (DEFAULT here).
    for head in (head_default, head_explicit):
      precisions = _dot_precisions(jax.make_jaxpr(head)(y).jaxpr, cfg.vocab_size)
      self.assertEqual(precisions, [(lax.Precision.DEFAULT, lax.Precision.DEFAULT)])
    np.testing.assert_array_equal(np.asarray(jax.jit(head_default)(y)), np.asarray(jax.jit(head_explicit)(y)))
    cfg_hi, _, _, head_hi = self._head(logits_dot_in_fp32=True, matmul_precision="highest")
    precisions = _dot_precisions(jax.make_jaxpr(head_hi)(y).jaxpr, cfg_hi.vocab_size)
    self.assertEqual(precisions, [(lax.Precision.HIGHEST, lax.Precision.HIGHEST)])

  def test_mtp_head_shares_the_precision(self):
    cfg, _, _, _ = self._head(logits_dot_in_fp32=True, logits_matmul_precision="highest")
    model = _build(cfg)
    self.assertEqual(model.decoder.logits_dense.matmul_precision, "highest")
    self.assertGreater(cfg.mtp_num_layers, 0)  # the MTP block calls decoder.apply_output_head (logits_dense)

  def test_tied_embedding_path(self):
    q = jnp.asarray(np.random.default_rng(2).standard_normal((2, 4, 32)), jnp.bfloat16)
    table = jnp.asarray(np.random.default_rng(3).standard_normal((64, 32)), jnp.float32)
    for prec, want in (("", None), ("highest", (lax.Precision.HIGHEST, lax.Precision.HIGHEST))):
      cfg = _config(logits_matmul_precision=prec)
      jaxpr = jax.make_jaxpr(lambda a, t, c=cfg: attend_on_embedding(a, t, jnp.float32, c))(q, table).jaxpr
      got = _dot_precisions(jaxpr, 64)
      self.assertEqual(len(got), 1)
      if want is None:
        self.assertIn(got[0], (None, (lax.Precision.DEFAULT, lax.Precision.DEFAULT)))
      else:
        self.assertEqual(got[0], want)


if __name__ == "__main__":
  unittest.main()
