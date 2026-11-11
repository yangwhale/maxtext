# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""log_indexer_loss_per_layer: off by default (aux unchanged); on, aux carries the per-layer indexer KD losses whose mean
is the logged indexer_loss, and the loss itself does not change."""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers.attention_mla import indexer_losses
from maxtext.trainers.pre_train import train as pre_train


class _Layer(nnx.Module):

  def __init__(self, value):
    self.value = value
    self.indexer_loss = indexer_losses(jnp.zeros(()))

  def __call__(self):
    self.indexer_loss = indexer_losses(jnp.asarray(self.value, jnp.float32))


class _MockDecoder(nnx.Module):
  """Two layers that each record an indexer KD loss, like the dense layer and the MTP layer of a Hy4 truncation."""

  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    self.mesh = jax.make_mesh((1, 1, 1, 1), ("data", "fsdp", "expert", "context"))
    self.a = _Layer(0.25)
    self.b = _Layer(0.75)

  def __call__(self, decoder_input_tokens, decoder_positions, decoder_segment_ids=None, encoder_images=None,
               encoder_image_masks=None, enable_dropout=False, decoder_target_tokens=None, decoder_target_mask=None):
    del decoder_positions, decoder_segment_ids, encoder_images, encoder_image_masks, enable_dropout
    del decoder_target_tokens, decoder_target_mask
    self.a()
    self.b()
    return jnp.zeros((*decoder_input_tokens.shape, self.vocab_size), dtype=jnp.float32)


def _config(**kw):
  argv = ["", "src/maxtext/configs/base.yml", "run_name=test_idx_per_layer", "attention_type=mla", "q_lora_rank=32", "attention=dot_product", "use_indexer=True",
          "indexer_loss_scaling_factor=1.0", "indexer_sparse_training=True", "max_target_length=16",
          "indexer_topk=2", "enable_checkpointing=False", "vocab_size=32"] + [f"{k}={v}" for k, v in kw.items()]
  return pyconfig.initialize(argv)


def _data():
  b, s = 2, 16
  return {
      "inputs": jnp.zeros((b, s), dtype=jnp.int32),
      "inputs_position": jnp.broadcast_to(jnp.arange(s), (b, s)),
      "inputs_segmentation": jnp.ones((b, s), dtype=jnp.int32),
      "targets": jnp.zeros((b, s), dtype=jnp.int32),
      "targets_segmentation": jnp.ones((b, s), dtype=jnp.int32),
  }


class IndexerLossPerLayerTest(unittest.TestCase):

  def test_off_by_default(self):
    cfg = _config()
    self.assertFalse(cfg.log_indexer_loss_per_layer)
    _, aux = pre_train.loss_fn(_MockDecoder(cfg.vocab_size), cfg, _data(), None, None, is_train=True)
    self.assertNotIn("indexer_loss_per_layer", aux)
    np.testing.assert_allclose(float(aux["indexer_loss"]), 0.5)

  def test_on_records_each_layer_and_keeps_loss(self):
    cfg_off, cfg_on = _config(), _config(log_indexer_loss_per_layer=True)
    loss_off, aux_off = pre_train.loss_fn(_MockDecoder(cfg_off.vocab_size), cfg_off, _data(), None, None, is_train=True)
    loss_on, aux_on = pre_train.loss_fn(_MockDecoder(cfg_on.vocab_size), cfg_on, _data(), None, None, is_train=True)
    np.testing.assert_array_equal(np.asarray(aux_on["indexer_loss_per_layer"]), np.array([0.25, 0.75], np.float32))
    self.assertEqual(float(aux_on["indexer_loss"]), float(np.mean(aux_on["indexer_loss_per_layer"])))
    self.assertEqual(float(loss_on), float(loss_off))
    self.assertEqual(float(aux_on["indexer_loss"]), float(aux_off["indexer_loss"]))


if __name__ == "__main__":
  unittest.main()
