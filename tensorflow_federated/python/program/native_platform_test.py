# Copyright 2022, The TensorFlow Federated Authors.
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

import asyncio
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import numpy as np
import tree

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.program import native_platform
from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import structure_utils


def _create_task(value: object) -> object:

  async def _fn(value: object) -> object:
    return value

  coro = _fn(value)
  return asyncio.create_task(coro)


def _create_identity_federated_computation(
    type_signature: federated_language.Type,
) -> federated_language.framework.Computation:
  @federated_language.federated_computation(type_signature)
  def _identity(value: object) -> object:
    return value

  return _identity


def _create_identity_tensorflow_computation(
    type_signature: federated_language.Type,
) -> federated_language.framework.Computation:
  @tensorflow_computation.tf_computation(type_signature)
  def _identity(value: object) -> object:
    return value

  return _identity


class NativeValueReferenceTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor_bool',
          lambda: _create_task(True),
          federated_language.TensorType(np.bool_),
          True,
      ),
      (
          'tensor_int',
          lambda: _create_task(1),
          federated_language.TensorType(np.int32),
          1,
      ),
      (
          'tensor_str',
          lambda: _create_task('abc'),
          federated_language.TensorType(np.str_),
          'abc',
      ),
      (
          'sequence',
          lambda: _create_task([1, 2, 3]),
          federated_language.SequenceType(np.int32),
          [1, 2, 3],
      ),
  )
  async def test_get_value_returns_value(
      self, task_factory, type_signature, expected_value
  ):
    task = task_factory()
    reference = native_platform.NativeValueReference(task, type_signature)

    actual_value = await reference.get_value()

    tree.assert_same_structure(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)


class CreateStructureOfReferencesTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor',
          lambda: _create_task(1),
          federated_language.TensorType(np.int32),
          lambda: native_platform.NativeValueReference(
              _create_task(1), federated_language.TensorType(np.int32)
          ),
      ),
      (
          'sequence',
          lambda: _create_task([1, 2, 3]),
          federated_language.SequenceType(np.int32),
          lambda: native_platform.NativeValueReference(
              _create_task([1, 2, 3]),
              federated_language.SequenceType(np.int32),
          ),
      ),
      (
          'federated_server',
          lambda: _create_task(1),
          federated_language.FederatedType(np.int32, federated_language.SERVER),
          lambda: native_platform.NativeValueReference(
              _create_task(1), federated_language.TensorType(np.int32)
          ),
      ),
      (
          'struct_unnamed',
          lambda: _create_task([True, 1, 'abc']),
          federated_language.StructWithPythonType(
              [np.bool_, np.int32, np.str_], list
          ),
          lambda: [
              native_platform.NativeValueReference(
                  _create_task(True), federated_language.TensorType(np.bool_)
              ),
              native_platform.NativeValueReference(
                  _create_task(1), federated_language.TensorType(np.int32)
              ),
              native_platform.NativeValueReference(
                  _create_task('abc'), federated_language.TensorType(np.str_)
              ),
          ],
      ),
      (
          'struct_named',
          lambda: _create_task({'a': True, 'b': 1, 'c': 'abc'}),
          federated_language.StructWithPythonType(
              [
                  ('a', np.bool_),
                  ('b', np.int32),
                  ('c', np.str_),
              ],
              dict,
          ),
          lambda: {
              'a': native_platform.NativeValueReference(
                  _create_task(True), federated_language.TensorType(np.bool_)
              ),
              'b': native_platform.NativeValueReference(
                  _create_task(1), federated_language.TensorType(np.int32)
              ),
              'c': native_platform.NativeValueReference(
                  _create_task('abc'), federated_language.TensorType(np.str_)
              ),
          },
      ),
      (
          'struct_nested',
          lambda: _create_task({'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}}),
          federated_language.StructWithPythonType(
              [
                  (
                      'x',
                      federated_language.StructWithPythonType(
                          [
                              ('a', np.bool_),
                              ('b', np.int32),
                          ],
                          dict,
                      ),
                  ),
                  (
                      'y',
                      federated_language.StructWithPythonType(
                          [
                              ('c', np.str_),
                          ],
                          dict,
                      ),
                  ),
              ],
              dict,
          ),
          lambda: {
              'x': {
                  'a': native_platform.NativeValueReference(
                      _create_task(True),
                      federated_language.TensorType(np.bool_),
                  ),
                  'b': native_platform.NativeValueReference(
                      _create_task(1),
                      federated_language.TensorType(np.int32),
                  ),
              },
              'y': {
                  'c': native_platform.NativeValueReference(
                      _create_task('abc'),
                      federated_language.TensorType(np.str_),
                  ),
              },
          },
      ),
  )
  async def test_returns_value(
      self, task_factory, type_signature, expected_value_factory
  ):
    task = task_factory()
    actual_value = native_platform._create_structure_of_references(
        task, type_signature
    )

    expected_value = expected_value_factory()
    actual_value = await federated_language.program.materialize_value(
        actual_value
    )
    expected_value = await federated_language.program.materialize_value(
        expected_value
    )
    tree.assert_same_structure(actual_value, expected_value)
    program_test_utils.assert_same_key_order(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'federated_clients',
          federated_language.FederatedType(
              np.int32, federated_language.CLIENTS
          ),
      ),
      ('function', federated_language.FunctionType(np.int32, np.int32)),
      ('placement', federated_language.PlacementType()),
  )
  async def test_raises_not_implemented_error_with_type_signature(
      self, type_signature
  ):
    task = _create_task(1)

    with self.assertRaises(NotImplementedError):
      native_platform._create_structure_of_references(task, type_signature)


class NativeFederatedContextTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      (
          'tensor',
          _create_identity_federated_computation(
              federated_language.TensorType(np.int32)
          ),
          1,
          1,
      ),
      (
          'sequence',
          _create_identity_tensorflow_computation(
              federated_language.SequenceType(np.int32)
          ),
          [1, 2, 3],
          [1, 2, 3],
      ),
      (
          'federated_server',
          _create_identity_federated_computation(
              federated_language.FederatedType(
                  np.int32, federated_language.SERVER
              )
          ),
          1,
          1,
      ),
      (
          'struct_unnamed',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [np.bool_, np.int32, np.str_], list
              )
          ),
          [True, 1, 'abc'],
          [True, 1, b'abc'],
      ),
      (
          'struct_named_ordered',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      ('a', np.bool_),
                      ('b', np.int32),
                      ('c', np.str_),
                  ],
                  dict,
              )
          ),
          {'a': True, 'b': 1, 'c': 'abc'},
          {'a': True, 'b': 1, 'c': b'abc'},
      ),
      (
          'struct_named_unordered',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      ('c', np.str_),
                      ('b', np.int32),
                      ('a', np.bool_),
                  ],
                  dict,
              )
          ),
          {'c': 'abc', 'b': 1, 'a': True},
          {'c': b'abc', 'b': 1, 'a': True},
      ),
  )
  async def test_invoke_returns_result(self, comp, arg, expected_value):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await federated_language.program.materialize_value(result)

    tree.assert_same_structure(actual_value, expected_value)
    program_test_utils.assert_same_key_order(actual_value, expected_value)
    actual_value = program_test_utils.to_python(actual_value)
    expected_value = program_test_utils.to_python(expected_value)
    self.assertEqual(actual_value, expected_value)

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          federated_language.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_partially_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', federated_language.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_sequentially(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=federated_language.framework.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      flattened = structure_utils.flatten(result)
      materialized = [await v.get_value() for v in flattened]
      actual_value = structure_utils.unflatten_as(result, materialized)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          federated_language.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_partially_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', federated_language.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_concurrently(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=federated_language.framework.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await federated_language.program.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  @parameterized.named_parameters(
      (
          'struct_nested',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      (
                          'y',
                          federated_language.StructWithPythonType(
                              [
                                  ('c', np.str_),
                              ],
                              dict,
                          ),
                      ),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {'c': 'abc'}},
          {'x': {'a': True, 'b': 1}, 'y': {'c': b'abc'}},
      ),
      (
          'struct_partially_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      (
                          'x',
                          federated_language.StructWithPythonType(
                              [
                                  ('a', np.bool_),
                                  ('b', np.int32),
                              ],
                              dict,
                          ),
                      ),
                      ('y', federated_language.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {'a': True, 'b': 1}, 'y': {}},
          {'x': {'a': True, 'b': 1}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_materialized_multiple(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=federated_language.framework.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await asyncio.gather(
          federated_language.program.materialize_value(result),
          federated_language.program.materialize_value(result),
          federated_language.program.materialize_value(result),
      )

    expected_value = [expected_value] * 3
    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_called_once()

  @parameterized.named_parameters(
      (
          'struct_unnamed_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType([], list)
          ),
          [],
          [],
      ),
      (
          'struct_named_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType([], dict)
          ),
          {},
          {},
      ),
      (
          'struct_nested_empty',
          _create_identity_federated_computation(
              federated_language.StructWithPythonType(
                  [
                      ('x', federated_language.StructWithPythonType([], dict)),
                      ('y', federated_language.StructWithPythonType([], dict)),
                  ],
                  dict,
              )
          ),
          {'x': {}, 'y': {}},
          {'x': {}, 'y': {}},
      ),
  )
  async def test_invoke_returns_result_comp_not_called(
      self, comp, arg, expected_value
  ):
    context = execution_contexts.create_async_local_cpp_execution_context()
    mock_context = mock.Mock(
        spec=federated_language.framework.AsyncExecutionContext, wraps=context
    )
    context = native_platform.NativeFederatedContext(mock_context)

    with program_test_utils.assert_not_warns(RuntimeWarning):
      result = context.invoke(comp, arg)
      actual_value = await federated_language.program.materialize_value(result)

    self.assertEqual(actual_value, expected_value)
    mock_context.invoke.assert_not_called()

  @parameterized.named_parameters(
      (
          'federated_clients',
          _create_identity_federated_computation(
              federated_language.FederatedType(
                  np.int32, federated_language.CLIENTS
              )
          ),
          1,
      ),
      (
          'function',
          _create_identity_federated_computation(
              federated_language.FunctionType(np.int32, np.int32)
          ),
          _create_identity_federated_computation(
              federated_language.TensorType(np.int32)
          ),
      ),
      (
          'placement',
          _create_identity_federated_computation(
              federated_language.PlacementType()
          ),
          None,
      ),
  )
  def test_invoke_raises_value_error_with_comp(self, comp, arg):
    context = execution_contexts.create_async_local_cpp_execution_context()
    context = native_platform.NativeFederatedContext(context)

    with self.assertRaises(ValueError):
      context.invoke(comp, arg)


if __name__ == '__main__':
  absltest.main()
