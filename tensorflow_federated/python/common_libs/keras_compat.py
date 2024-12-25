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
import keras
import tf_keras
import keras.src.backend.common as keras_common
from tf_keras.src.optimizers.legacy import optimizer_v2
import tensorflow as tf
from typing import Union

Model = Union[tf_keras.Model, keras.Model]


def is_compiled(model: Model):
    """Checks that `model` is of compiled.

    Args:
      model: An instance of a tf_keras.Model or a keras.Model.

    Returns:
      True if the model is compiled; False otherwise.

    Raises:
      TypeError: when the model is not an instance of tf_keras.Model or a keras.Model.
    """
    if isinstance(model, tf_keras.Model):
        return model._is_compiled
    elif isinstance(model, keras.Model):
        return model.compiled
    else:
        raise TypeError(
            'Expected an instance of a tf_keras.Model, or a '
            'keras.Model; found a model of type '
            f'{type(model)}'
        )


def keras_dtype_to_tf(dtype_str):
    """Converts the dtype from its string representation to an
        equivalent TensorFlow representation.

    Args:
      dtype_str: A string dtype representation or a dtype.

    Returns:
      The tf.dtype which matches the string representation,
        if dtype_str is a string, otherwise returns dtype-str.
    """
    if not isinstance(dtype_str, str):
        return dtype_str
    return {
        "float16": tf.float16,
        "float32": tf.float32,
        "float64": tf.float64,
        "uint8": tf.uint8,
        "uint16": tf.uint16,
        "uint32": tf.uint32,
        "uint64": tf.uint64,
        "int8": tf.int8,
        "int16": tf.int16,
        "int32": tf.int32,
        "int64": tf.int64,
        "bfloat16": tf.bfloat16,
        "bool": tf.bool,
        "string": tf.string,
        "float8_e4m3fn": tf.dtypes.experimental.float8_e4m3fn,
        "float8_e5m2": tf.dtypes.experimental.float8_e5m2
    }.get(dtype_str, tf.float32)


#TODO Add additional checks
def is_keras3(obj: object):
    """Checks if the 'obj' is from the keras 3 library classes.

    Args:
      obj: An instance of one of the keras 2 or keras 3 classes.

    Returns:
      True if the object is from the keras 3 library; False otherwise.
    """
    return isinstance(obj, (keras.Model, keras.Variable, keras.Metric, keras.Layer, keras.Loss, keras.Optimizer, keras_common.KerasVariable))


def get_optimizer_variables(optimizer: Union[tf_keras.optimizers.Optimizer, keras.optimizers.Optimizer, optimizer_v2.OptimizerV2]):
  """
    Retrieves the variables from an optimizer.

    If the optimizer's `variables` attribute is a callable function,
    it calls this function to get variables. Otherwise, it returns the `variables` attribute directly.

    Args:
      optimizer: An instance of a tf_keras optimizer or a keras optimizer.

    Returns:
      List of variables associated with the optimizer.

    Raises:
      TypeError: If optimizer is not an instance of tf_keras.optimizers.Optimizer,
       optimizer_v2.OptimizerV2 or keras.optimizers.Optimizer.
    """
  if isinstance(optimizer, (tf_keras.optimizers.Optimizer, keras.optimizers.Optimizer, optimizer_v2.OptimizerV2)):
    if callable(optimizer.variables):
      return optimizer.variables()
    return optimizer.variables
  else:
    raise TypeError(
      'Expected an instance of a tf_keras.optimizers.Optimizer, or a '
      'keras.optimizers.Optimizer; found an optimizer of type '
      f'{type(optimizer)}'
    )


def ref(variable: Union[tf.Variable, keras.Variable, keras_common.KerasVariable]):
  """
  Returns the reference/id of a Keras or TensorFlow variable.

  Args:
    variable: An instance of a tf.Variable, a keras.backend.common.KerasVariable
     or a keras.Variable.

  Raises:
    TypeError: if the variable is not an instance of keras.Variable,
     keras.backend.common.KerasVariable or tf.Variable.

  Returns:
    Reference if the variable is from the tf_keras library,
    otherwise returns the id of the variable.
  """
  if isinstance(variable, tf.Variable):
    return variable.ref()
  elif isinstance(variable, (keras.Variable, keras_common.KerasVariable)):
    return id(variable)
  else:
    raise TypeError(f'Expected keras.Variable, keras.backend.common.KerasVariable or tf.Variable, but got {type(variable)}')


def get_variable(variable: Union[tf.Variable, keras.Variable, keras_common.KerasVariable]):
  """
  Returns the variable value of a Keras or TensorFlow variable.

  Args:
    variable: An instance of a tf.Variable, a keras.backend.common.KerasVariable
     or a keras.Variable.

  Returns:
    Value of the variable if it is from the keras 3 library,
    otherwise returns the variable.
  """
  if isinstance(variable, (keras.Variable, keras_common.KerasVariable)):
    return variable.value
  else:
    return variable


def get_variables(variables: Union[list, tuple]):
  """
  Returns the variable value of a Keras or TensorFlow variable.

  Args:
    variables: A list or a tuple of instances of tf.Variable, keras.backend.common.KerasVariable
     or keras.Variable.

  Returns:
    Value of the variables if they are from the keras 3 library,
    otherwise returns the variables.
  """
  return [get_variable(var) for var in variables]


def clone_model(model: Union[tf_keras.Model, keras.Model]):
  """
  Clones a tf_keras.Model or a keras.Model and returns the cloned model.

  Args:
    model: A tf_keras.Model or a keras.Model.

  Returns:
    An instance of `Model` reproducing the behavior
    of the original model, on top of new inputs tensors,
    using newly instantiated weights. The cloned model may behave
    differently from the original model if a custom `clone_function`
    or `call_function` modifies a layer or layer call.
  """
  if is_keras3(model):
    return keras.models.clone_model(model)
  else:
    return tf_keras.models.clone_model(model)


def int_shape(x):
  """Returns shape of tensor/variable as a tuple of int/None entries.

  Args:
      x: Tensor or variable.

  Returns:
      A tuple of integers (or None entries).

  """
  try:
    shape = x.shape
    if not isinstance(shape, tuple):
      shape = tuple(shape.as_list())
    return shape
  except ValueError:
    return None


def restructure_collection(collection, target_collection):
  """Restructures the collection to the same type as the target collection

  Args:
      collection: Collection that should be restructured
      target_collection: Target collection with the target collection type

  Returns:
      The restructured collection if the target collection is of a different
      type than the original collection, else returns the original collection

  """
  return keras.tree.pack_sequence_as([0] * len(collection), collection) \
    if isinstance(target_collection, list) else collection
