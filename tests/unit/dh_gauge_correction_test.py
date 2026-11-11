"""lm_head_dh_gauge_scale: the correction term is zero in the forward pass and, in the backward pass, changes only
the hidden-state gradient by (scale - 1) * rowsum(dlogits) * mean_vocab(kernel)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers.nnx_decoders import _dh_gauge_correction, _dh_remove_direction


def _loss(y, kernel, labels, scale):
  logits = jnp.einsum("bld,dv->blv", y, kernel)
  if scale != 1.0:
    logits = logits + _dh_gauge_correction(y, jax.lax.stop_gradient(jnp.mean(kernel, axis=-1)), logits, scale)
  logp = jax.nn.log_softmax(logits, axis=-1)
  # a deliberately non-centered cotangent: weight the target term so the row sum of dlogits is nonzero
  return -jnp.sum(jnp.take_along_axis(logp, labels[..., None], axis=-1)) + 0.1 * jnp.sum(logits**2)


class DhGaugeCorrectionTest(unittest.TestCase):

  def test_forward_unchanged_and_backward(self):
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((2, 3, 8)), jnp.float32)
    kernel = jnp.asarray(rng.standard_normal((8, 16)), jnp.float32)
    labels = jnp.asarray(rng.integers(0, 16, (2, 3)))
    np.testing.assert_array_equal(_loss(y, kernel, labels, 1.0), _loss(y, kernel, labels, 0.0))
    gy1, gk1 = jax.grad(_loss, argnums=(0, 1))(y, kernel, labels, 1.0)
    for scale in (0.0, 2.0):
      gy, gk = jax.grad(_loss, argnums=(0, 1))(y, kernel, labels, scale)
      np.testing.assert_allclose(gk, gk1, rtol=1e-6)
      logits = jnp.einsum("bld,dv->blv", y, kernel)
      dlog = jax.grad(lambda lg: -jnp.sum(jnp.take_along_axis(jax.nn.log_softmax(lg, -1), labels[..., None], -1))
                      + 0.1 * jnp.sum(lg**2))(logits)
      s = jnp.sum(dlog, axis=-1, keepdims=True)
      np.testing.assert_allclose(gy, gy1 + (scale - 1.0) * s * jnp.mean(kernel, axis=-1), rtol=1e-5, atol=1e-5)
    # scale 0: the corrected hidden gradient equals the one from the row-centered cotangent
    gy0 = jax.grad(_loss)(y, kernel, labels, 0.0)
    dlog_c = dlog - jnp.mean(dlog, axis=-1, keepdims=True)
    np.testing.assert_allclose(gy0, jnp.einsum("blv,dv->bld", dlog_c, kernel), rtol=1e-5, atol=1e-5)


  def test_remove_direction(self):
    rng = np.random.default_rng(1)
    y = jnp.asarray(rng.standard_normal((2, 3, 8)), jnp.float32)
    u = jnp.asarray(rng.standard_normal(8), jnp.float32)
    u = u / jnp.linalg.norm(u)
    c = jnp.asarray(rng.standard_normal((2, 3, 8)), jnp.float32)
    f = lambda yy, a: jnp.sum(c * jnp.sin(_dh_remove_direction(yy, u, a)))  # noqa: E731
    np.testing.assert_array_equal(f(y, 0.5), jnp.sum(c * jnp.sin(y)))
    g0 = jax.grad(f)(y, 0.0)
    for a in (0.5, 1.0):
      g = jax.grad(f)(y, a)
      np.testing.assert_allclose(g, g0 - a * jnp.sum(g0 * u, -1, keepdims=True) * u, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
