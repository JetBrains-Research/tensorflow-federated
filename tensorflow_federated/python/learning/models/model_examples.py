# Copyright 2018, The TensorFlow Federated Authors.
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
"""Simple examples implementing the Model interface."""

import collections
from collections.abc import Callable
import functools
from typing import Union

import tensorflow as tf
import tf_keras
import keras

from tensorflow_federated.python.learning.models import variable


class LinearRegression(variable.VariableModel):
  """Example of a simple linear regression implemented directly."""

  def __init__(self, feature_dim: int = 2, has_unconnected: bool = False):
    # Define all the variables, similar to what Keras Layers and Models
    # do in build().
    self._feature_dim = feature_dim
    # TODO: b/124070381 - Support for integers in num_examples, etc., is handled
    # here in learning, by adding an explicit cast to a float where necessary in
    # order to pass typechecking in the reference executor.
    self._num_examples = tf.Variable(0, trainable=False)
    self._num_batches = tf.Variable(0, trainable=False)
    self._loss_sum = tf.Variable(0.0, trainable=False)
    self._a = tf.Variable([[0.0]] * feature_dim, trainable=True)
    self._b = tf.Variable(0.0, trainable=True)
    # Define a non-trainable model variable (another bias term) for code
    # coverage in testing.
    self._c = tf.Variable(0.0, trainable=False)
    self._input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, self._feature_dim], tf.float32),
        y=tf.TensorSpec([None, 1], tf.float32),
    )
    self.has_unconnected = has_unconnected
    if has_unconnected:
      self._unconnected = tf.Variable(0.0, trainable=True)

  @property
  def trainable_variables(self) -> list[tf.Variable]:
    if self.has_unconnected:
      return [self._a, self._b, self._unconnected]
    return [self._a, self._b]

  @property
  def non_trainable_variables(self) -> list[tf.Variable]:
    return [self._c]

  @property
  def local_variables(self) -> list[tf.Variable]:
    return [self._num_examples, self._num_batches, self._loss_sum]

  @property
  def input_spec(self):
    # Model expects batched input, but the batch dimension is unspecified.
    return self._input_spec

  @tf.function
  def predict_on_batch(self, x, training=True):
    del training  # Unused.
    return tf.matmul(x, self._a) + self._b + self._c

  @tf.function
  def forward_pass(self, batch_input, training=True) -> variable.BatchOutput:
    if not self._input_spec['y'].is_compatible_with(batch_input['y']):
      raise ValueError(
          "Expected batch_input['y'] to be compatible with "
          f"{self._input_spec['y']} but found {batch_input['y']}"
      )
    if not self._input_spec['x'].is_compatible_with(batch_input['x']):
      raise ValueError(
          "Expected batch_input['x'] to be compatible with "
          "{self._input_spec['x']} but found {batch_input['x']}"
      )
    predictions = self.predict_on_batch(x=batch_input['x'], training=training)
    residuals = predictions - batch_input['y']
    num_examples = tf.gather(tf.shape(predictions), 0)
    total_loss = 0.5 * tf.reduce_sum(tf.pow(residuals, 2))

    self._loss_sum.assign_add(total_loss)
    self._num_examples.assign_add(num_examples)
    self._num_batches.assign_add(1)

    average_loss = total_loss / tf.cast(num_examples, tf.float32)
    return variable.BatchOutput(
        loss=average_loss, predictions=predictions, num_examples=num_examples
    )

  @tf.function
  def report_local_unfinalized_metrics(
      self,
  ) -> collections.OrderedDict[str, Union[tf.Tensor, list[tf.Tensor]]]:
    """Creates an `collections.OrderedDict` of metric names to unfinalized values.

    Returns:
      An `collections.OrderedDict` of metric names to unfinalized values. The
      `collections.OrderedDict`
      has the same keys (metric names) as the `collections.OrderedDict` returned
      by the
      method `metric_finalizers()`, and can be used as input to the finalizers
      to get the finalized metric values. This method and `metric_finalizers()`
      method can be used together to build a cross-client metrics aggregator
      when defining the federated training processes or evaluation computations.
    """
    return collections.OrderedDict(
        loss=[self._loss_sum, tf.cast(self._num_examples, tf.float32)],
        num_examples=self._num_examples,
    )

  def metric_finalizers(
      self,
  ) -> collections.OrderedDict[
      str, Callable[[Union[tf.Tensor, list[tf.Tensor]]], tf.Tensor]
  ]:
    """Creates an `collections.OrderedDict` of metric names to finalizers.

    Returns:
      An `collections.OrderedDict` of metric names to finalizers. A finalizer is
      a
      `tf.function` decorated callable that takes in a metric's unfinalized
      values (returned by `report_local_unfinalized_metrics()`), and returns the
      finalized values. This method and the `report_local_unfinalized_metrics()`
      method can be used together to construct a cross-client metrics aggregator
      when defining the federated training processes or evaluation computations.
    """
    return collections.OrderedDict(
        loss=tf.function(func=lambda x: x[0] / x[1]),
        num_examples=tf.function(func=lambda x: x),
    )

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))


def _dense_all_zeros_layer(input_dims=None, output_dim=1):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.

  Returns:
    a `tf_keras.layers.Dense` object.
  """
  build_keras_dense_layer = functools.partial(
      tf_keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      activation=None,
  )
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_zeros_layer_keras3(input_dims=None, output_dim=1):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.

  Returns:
    a `keras.layers.Dense` object.
  """

  build_keras_dense_layer = functools.partial(
      keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      activation=None,
  )
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_zeros_regularized_layer(
    input_dims=None, output_dim=1, regularization_constant=0.01
):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `tf_keras.layers.Dense` object.
  """
  regularizer = tf_keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      tf_keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None,
  )
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_zeros_regularized_layer_keras3(
    input_dims=None, output_dim=1, regularization_constant=0.01
):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `keras.layers.Dense` object.
  """
  regularizer = keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None,
  )
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_ones_regularized_layer(
    input_dims=None, output_dim=1, regularization_constant=0.01
):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to ones. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `tf_keras.layers.Dense` object.
  """
  regularizer = tf_keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      tf_keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='ones',
      bias_initializer='ones',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None,
  )
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_ones_regularized_layer_keras3(
    input_dims=None, output_dim=1, regularization_constant=0.01
):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to ones. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `tf_keras.layers.Dense` object.
  """
  regularizer = keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='ones',
      bias_initializer='ones',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None,
  )

  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def build_linear_regression_keras_sequential_model(feature_dims=2):
  """Build a linear regression `tf_keras.Model` using the Sequential API."""
  keras_model = tf_keras.models.Sequential()
  keras_model.add(_dense_all_zeros_layer(input_dims=feature_dims))
  return keras_model


def build_linear_regression_keras3_sequential_model(feature_dims=2):
  """Build a linear regression `keras.Model` using the Sequential API."""
  keras_model = keras.models.Sequential()
  keras_model.add(_dense_all_zeros_layer_keras3(input_dims=feature_dims))
  return keras_model


def build_linear_regression_regularized_keras_sequential_model(
    feature_dims=2, regularization_constant=0.01
):
  """Build a linear regression `tf_keras.Model` using the Sequential API."""
  keras_model = tf_keras.models.Sequential()
  keras_model.add(
      _dense_all_zeros_regularized_layer(
          feature_dims, regularization_constant=regularization_constant
      )
  )
  return keras_model


def build_linear_regression_regularized_keras3_sequential_model(
    feature_dims=2, regularization_constant=0.01
):
  """Build a linear regression `keras.Model` using the Sequential API."""
  keras_model = keras.models.Sequential()
  keras_model.add(
      _dense_all_zeros_regularized_layer_keras3(
          feature_dims, regularization_constant=regularization_constant
      )
  )
  return keras_model


def build_linear_regression_ones_regularized_keras_sequential_model(
    feature_dims=2, regularization_constant=0.01
):
  """Build a linear regression `tf_keras.Model` using the Sequential API."""
  keras_model = tf_keras.models.Sequential()
  keras_model.add(
      _dense_all_ones_regularized_layer(
          feature_dims, regularization_constant=regularization_constant
      )
  )
  return keras_model


def build_linear_regression_ones_regularized_keras3_sequential_model(
    feature_dims=2, regularization_constant=0.01
):
  """Build a linear regression `keras.Model` using the Sequential API."""
  keras_model = keras.models.Sequential()
  keras_model.add(
      _dense_all_ones_regularized_layer_keras3(
          feature_dims, regularization_constant=regularization_constant
      )
  )
  return keras_model


def build_linear_regression_keras_functional_model(feature_dims=2):
  """Build a linear regression `tf_keras.Model` using the functional API."""
  a = tf_keras.layers.Input(shape=(feature_dims,), dtype=tf.float32)
  b = _dense_all_zeros_layer()(a)
  return tf_keras.Model(inputs=a, outputs=b)


def build_linear_regression_keras3_functional_model(feature_dims=2):
  """Build a linear regression `keras.Model` using the functional API."""
  a = keras.layers.Input(shape=(feature_dims,), dtype=tf.float32)
  b = _dense_all_zeros_layer_keras3(None)(a)
  return keras.Model(inputs=[a], outputs=[b])


def build_linear_regression_keras_subclass_model(feature_dims=2):
  """Build a linear regression model by sub-classing `tf_keras.Model`."""
  del feature_dims  # Unused.

  class _KerasLinearRegression(tf_keras.Model):

    def __init__(self):
      super().__init__()
      self._weights = _dense_all_zeros_layer()

    def call(self, inputs, training=None, mask=None):
      del training, mask  # Unused.
      return self._weights(inputs)

  return _KerasLinearRegression()


def build_embedding_keras_model(vocab_size=10):
  """Builds a test model with an embedding initialized to one-hot vectors."""
  keras_model = tf_keras.models.Sequential()
  keras_model.add(tf_keras.layers.Embedding(input_dim=vocab_size, output_dim=5,
                                            embeddings_initializer=keras.initializers.RandomUniform(seed=42)))
  keras_model.add(tf_keras.layers.Softmax())
  return keras_model

def build_embedding_keras3_model(vocab_size=10):
  """Builds a test model with an embedding initialized to one-hot vectors."""
  keras_model = keras.models.Sequential()
  keras_model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=5,
                                         embeddings_initializer=keras.initializers.RandomUniform(seed=42)))
  keras_model.add(keras.layers.Softmax())
  return keras_model


def build_conv_batch_norm_keras_model():
  """Builds a test model with convolution and batch normalization."""
  # This is an example of a model that has trainable and non-trainable
  # variables.
  l = tf_keras.layers
  data_format = 'channels_last'
  max_pool = l.MaxPooling2D(
      pool_size=(2, 2), strides=(2, 2), padding='same', data_format=data_format
  )
  keras_model = tf_keras.models.Sequential([
      l.Reshape(target_shape=[28, 28, 1], input_shape=(28 * 28,)),
      l.Conv2D(
          filters=32,
          kernel_size=5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      max_pool,
      l.BatchNormalization(),
      l.Conv2D(
          filters=64,
          kernel_size=5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      max_pool,
      l.BatchNormalization(),
      l.Flatten(),
      l.Dense(
          units=1024,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      l.Dropout(rate=0.4),
      l.Dense(10, kernel_initializer='zeros', bias_initializer='zeros'),
  ])
  return keras_model


def build_conv_batch_norm_keras3_model():
  """Builds a test model with convolution and batch normalization."""
  # This is an example of a model that has trainable and non-trainable
  # variables.
  l = keras.layers
  data_format = 'channels_last'
  max_pool = l.MaxPooling2D(
      pool_size=(2, 2), strides=(2, 2), padding='same', data_format=data_format
  )
  keras_model = keras.models.Sequential([
      l.Input(shape=(28 * 28,)),
      l.Reshape(target_shape=[28, 28, 1]),
      l.Conv2D(
          filters=32,
          kernel_size=5,
          padding='same',
          data_format=data_format,
          activation='relu',
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      max_pool,
      l.BatchNormalization(),
      l.Conv2D(
          filters=64,
          kernel_size=5,
          padding='same',
          data_format=data_format,
          activation='relu',
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      max_pool,
      l.BatchNormalization(),
      l.Flatten(),
      l.Dense(
          units=1024,
          activation='relu',
          kernel_initializer='zeros',
          bias_initializer='zeros',
      ),
      l.Dropout(rate=0.4),
      l.Dense(10, kernel_initializer='zeros', bias_initializer='zeros'),
  ])
  return keras_model


def build_multiple_inputs_keras_model():
  """Builds a test model with two inputs."""
  l = tf_keras.layers
  a = l.Input((1,), name='a')
  b = l.Input((1,), name='b')
  # Each input has a single, independent dense layer, which are combined into
  # a final dense layer.
  output = l.Dense(units=1)(
      l.concatenate([
          l.Dense(units=1)(a),
          l.Dense(units=1)(b),
      ])
  )
  return tf_keras.Model(inputs={'a': a, 'b': b}, outputs=[output])


def build_multiple_inputs_keras3_model():
  """Builds a test model with two inputs."""
  l = keras.layers
  a = l.Input((1,), name='a')
  b = l.Input((1,), name='b')
  # Each input has a single, independent dense layer, which are combined into
  # a final dense layer.
  output = l.Dense(units=1)(
      l.concatenate([
          l.Dense(units=1)(a),
          l.Dense(units=1)(b),
      ])
  )
  return keras.Model(inputs={'a': a, 'b': b}, outputs=[output])


def build_multiple_outputs_keras_model():
  """Builds a test model with three outputs."""
  l = tf_keras.layers
  a = l.Input((1,))
  b = l.Input((1,))

  output_a = l.Dense(1)(a)
  output_b = l.Dense(1)(b)
  output_c = l.Dense(1)(l.concatenate([l.Dense(1)(a), l.Dense(1)(b)]))

  return tf_keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_multiple_outputs_keras3_model():
  """Builds a test model with three outputs."""
  l = keras.layers
  a = l.Input((1,))
  b = l.Input((1,))

  output_a = l.Dense(1)(a)
  output_b = l.Dense(1)(b)
  output_c = l.Dense(1)(l.concatenate([l.Dense(1)(a), l.Dense(1)(b)]))

  return keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_tupled_dict_outputs_keras_model():
  """Builds a test model with three outputs."""
  l = tf_keras.layers
  a = l.Input((1,))
  b = l.Input((1,))

  output_a = l.Dense(1)(a)
  output_b = l.Dense(1)(b)

  return tf_keras.Model(
      inputs=[a, b],
      outputs=({'output_1': output_a}, {'output_1': output_b}),
  )


def build_tupled_dict_outputs_keras3_model():
  """Builds a test model with three outputs."""
  l = keras.layers
  a = l.Input((1,))
  b = l.Input((1,))

  output_a = l.Dense(1)(a)
  output_b = l.Dense(1)(b)

  return keras.Model(
      inputs=[a, b],
      outputs=({'output_1': output_a}, {'output_1': output_b}),
  )


def build_multiple_outputs_regularized_keras_model(
    regularization_constant=0.01,
):
  """Builds a test model with three outputs.

  All weights are initialized to ones.

  Args:
    regularization_constant: L2 scaling constant (lambda) for all weights and
      biases.

  Returns:
    a `tf_keras.Model` object.
  """
  dense = functools.partial(
      _dense_all_ones_regularized_layer,
      output_dim=1,
      regularization_constant=regularization_constant,
  )
  a = tf_keras.layers.Input((1,))
  b = tf_keras.layers.Input((1,))

  output_a = dense()(a)
  output_b = dense()(b)
  output_c = dense()(tf_keras.layers.concatenate([dense()(a), dense()(b)]))

  return tf_keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_multiple_outputs_regularized_keras3_model(
    regularization_constant=0.01,
):
  """Builds a test model with three outputs.

  All weights are initialized to ones.

  Args:
    regularization_constant: L2 scaling constant (lambda) for all weights and
      biases.

  Returns:
    a `keras.Model` object.
  """
  dense = functools.partial(
      _dense_all_ones_regularized_layer_keras3,
      output_dim=1,
      regularization_constant=regularization_constant
  )
  a = keras.layers.Input((1,))
  b = keras.layers.Input((1,))

  output_a = dense()(a)
  output_b = dense()(b)
  output_c = dense()(keras.layers.concatenate([dense()(a), dense()(b)]))

  return keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_lookup_table_keras_model():
  """Builds a test model with embedding feature columns."""
  l = tf_keras.layers
  a = l.Input(shape=(1,), dtype=tf.string)
  # pylint: disable=g-deprecated-tf-checker
  embedded_lookup_feature = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='colors', vocabulary_list=('R', 'G', 'B')
      ),
      dimension=16,
  )
  # pylint: enable=g-deprecated-tf-checker
  dense_features = l.DenseFeatures([embedded_lookup_feature])({'colors': a})
  output = l.Dense(1)(dense_features)
  return tf_keras.Model(inputs=[a], outputs=[output])


def build_lookup_table_keras3_model():
    """Builds a test model with embedding feature columns."""

    color_to_int_map = {'R': 0, 'G': 1, 'B': 2}

    model = keras.models.Sequential()

    model.add(keras.layers.StringLookup(
        vocabulary=list(color_to_int_map.keys()), mask_token=None, num_oov_indices=0
    ))

    model.add(keras.layers.Embedding(input_dim=3, output_dim=16, input_length=1))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1))

    return model


def build_preprocessing_lookup_keras_model():
  """Builds a test model using processing layers."""
  l = tf_keras.layers
  a = l.Input(shape=(1,), dtype=tf.string)
  encoded = l.experimental.preprocessing.StringLookup(vocabulary=['A', 'B'])(a)
  return tf_keras.Model(inputs=[a], outputs=[encoded])


def build_preprocessing_lookup_keras3_model():
  """Builds a test model using processing layers."""
  l = keras.layers
  a = l.Input(shape=(1,), dtype=tf.string)
  encoded = l.StringLookup(vocabulary=['A', 'B'])(a)
  return tf_keras.Model(inputs=[a], outputs=[encoded])


def build_ragged_tensor_input_keras_model():
  """Builds a test model with ragged tensors as input."""
  return tf_keras.Sequential([
      tf_keras.layers.Input(shape=[None], dtype=tf.int64, ragged=True),
      tf_keras.layers.Embedding(1000, 16),
      tf_keras.layers.LSTM(units=32, use_bias=False),
      tf_keras.layers.Dense(units=32),
      tf_keras.layers.Activation(activation=tf.nn.relu),
      tf_keras.layers.Dense(units=1),
  ])

def build_ragged_tensor_input_keras3_model():
  """Builds a test model with ragged tensors as input."""
  return keras.Sequential([
      keras.layers.Input(shape=[None], dtype=tf.int64),
      keras.layers.Embedding(1000, 16, mask_zero=True),
      keras.layers.Masking(),
      keras.layers.LSTM(units=32, use_bias=False),
      keras.layers.Dense(units=32),
      keras.layers.Activation(activation=tf.nn.relu),
      keras.layers.Dense(units=1),
  ])
