# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for testing optimizers."""

from collections.abc import Callable, Collection

import tensorflow as tf
import tf_keras

from tensorflow_federated.python.learning.optimizers import optimizer as optimizer_base

# Positive-definite matrix with maximal eigenvalue approximately 13.3 and
# condition number approximately 260, generated by:
# np.random.seed(42)
# X = np.random.randn(5, 5)
# A = np.matmul(X, np.transpose(X))
_TEST_MATRIX = [
    [3.05979016, -0.67965499, -2.51916374, -0.98797549, -1.23950244],
    [-0.67965499, 3.65246689, -0.47892836, -1.56662479, -0.27436542],
    [-2.51916374, -0.47892836, 7.12618873, 4.98172794, 3.10724048],
    [-0.98797549, -1.56662479, 4.98172794, 4.259855, 1.48831719],
    [-1.23950244, -0.27436542, 3.10724048, 1.48831719, 4.52992126],
]

_INITIAL_W = [
    [-0.60170661],
    [1.85227818],
    [-0.01349722],
    [-1.05771093],
    [0.82254491],
]


def test_quadratic_problem():
  """Returns a test problem.

  The test problem is a 5-dimensional quadratic problem of form
  `(1/2) * w^t * A * w` with the optimum in w* = (0.0, 0.0, 0.0, 0.0, 0.0). The
  matrix A is positive-definite with maximal eigenvalue approximately 13.3 and
  condition number approximately 260.

  It is meant to be used as a simple deterministic problem to be used when unit
  testing optimizers.

  Returns:
    A tuple (initial_w, f, grad_w), where these are functions. First is a no-arg
    callable for creating the initial point w for optimization and others are
    one-arg callables returning the function value and gradient given a point w,
    respectively.
  """

  @tf.function
  def fn(w):
    a = tf.constant(_TEST_MATRIX, tf.float32)
    return tf.matmul(tf.matmul(w, a, transpose_a=True), w) / 2

  @tf.function
  def grad_fn(w):
    a = tf.constant(_TEST_MATRIX, tf.float32)
    return tf.matmul(a, w)

  @tf.function
  def initial_w():
    return tf.constant(_INITIAL_W, tf.float32)

  return initial_w, fn, grad_fn


class TestCase(tf.test.TestCase):
  """A helper class to test TFF optimizers."""

  def assert_optimizers_numerically_close(
      self,
      model_variables_fn: Callable[[], Collection[tf.Variable]],
      gradients: Collection[Collection[tf.Tensor]],
      tff_optimizer_fn: Callable[[], optimizer_base.Optimizer],
      keras_optimizer_fn: Callable[[], tf_keras.optimizers.Optimizer],
  ):
    """Test the numerical correctness of TFF optimizer by comparign to Keras.

    When implementing a `tff.learning.optimizer.Optimizer` that exists in Keras,
    this helper function can help compare the numerical results of the two
    optimizers given initial weight and a sequence of gradients. This is
    intended to improve the coverage of optimizer tests.

    Args:
      model_variables_fn: A no-arg function returns the intial model weights.
      gradients: A sequence of gradients for updating the model weights.
      tff_optimizer_fn: A no-arg function returns a
        `tff.learning.optimizer.Optimizer`.
      keras_optimizer_fn:  A no-arg function returns a
        `tf_keras.optimizers.Optimizer`.
    """

    def _run_tff():
      model_weights = model_variables_fn()
      optimizer = tff_optimizer_fn()
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_weights
      )
      state = optimizer.initialize(model_weight_specs)
      for grad in gradients:
        state, model_weights = optimizer.next(state, model_weights, grad)
      return model_weights

    def _run_keras():
      model_variables = model_variables_fn()
      optimizer = keras_optimizer_fn()
      for grad in gradients:
        optimizer.apply_gradients(zip(grad, model_variables))
      return model_variables

    self.assertAllClose(_run_tff(), _run_keras(), rtol=5e-5, atol=5e-5)
