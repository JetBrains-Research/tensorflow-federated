/* Copyright 2022, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/eager_computation.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/no_op.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace {

// TODO: b/256948367 - Move these common methods to a base test utility file.
template <class TfOp>
inline federated_language::TensorFlow::Binding TensorB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  federated_language::TensorFlow::Binding binding;
  *binding.mutable_tensor()->mutable_tensor_name() = node->name();
  return binding;
}

inline federated_language::TensorFlow::Binding StructB(
    const absl::Span<const federated_language::TensorFlow::Binding> elements) {
  federated_language::TensorFlow::Binding binding;
  auto struct_mut = binding.mutable_struct_();
  for (const auto& element : elements) {
    *struct_mut->add_element() = element;
  }
  return binding;
}

inline federated_language::Computation ComputationV(
    const tensorflow::Scope& scope,
    std::optional<federated_language::TensorFlow::Binding> in_binding,
    federated_language::TensorFlow::Binding out_binding,
    const std::optional<const tensorflow::Operation>& init_op = std::nullopt,
    const std::vector<tensorflow::FunctionDef> function_defs = {}) {
  federated_language::Computation comp_pb;
  federated_language::TensorFlow* tensorflow_pb = comp_pb.mutable_tensorflow();

  tensorflow::GraphDef graphdef_pb;

  absl::Status status = scope.ToGraphDef(&graphdef_pb);
  CHECK(status.ok()) << status;

  if (!function_defs.empty()) {
    for (const auto& f : function_defs) {
      *graphdef_pb.mutable_library()->add_function() = f;
    }
  }

  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  if (in_binding.has_value()) {
    *tensorflow_pb->mutable_parameter() = in_binding.value();
  }
  *tensorflow_pb->mutable_result() = out_binding;
  if (init_op.has_value()) {
    *tensorflow_pb->mutable_initialize_op() = init_op.value().node()->name();
  }
  return comp_pb;
}

TF_Tensor* FloatTensor(float v) {
  const int num_bytes = sizeof(float);
  float* values = new float[1];
  values[0] = v;
  return TF_NewTensor(
      TF_FLOAT, nullptr, 0, values, num_bytes,
      [](void* data, size_t, void*) { delete[] static_cast<float*>(data); },
      nullptr);
}

class EagerComputationTest : public ::testing::Test {};

TEST_F(EagerComputationTest, CallNoArgOneOutWithInitialize) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);

  auto fn = ComputationV(root, /*in_binding=*/std::nullopt,
                         /*out_binding=*/TensorB(read_var),
                         /*init_op=*/var_init);
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[3];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallNoArgOneOutWithGroupedInitialize) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);

  auto grouped_initializer =
      tensorflow::ops::NoOp(root.WithOpName("grouped_initializer")
                                .WithControlDependencies({var_init}));
  auto fn = ComputationV(root, /*in_binding=*/std::nullopt,
                         /*out_binding=*/TensorB(read_var),
                         /*init_op=*/grouped_initializer);
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[3];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallAdd) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  tensorflow::ops::AddV2 out(root, x, y);
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}), TensorB(out));
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);
  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallAddExtraPlaceholder) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  // Not consumed placeholder. It should be ignored.
  tensorflow::ops::Placeholder z(root, tensorflow::DT_FLOAT);
  tensorflow::ops::AddV2 out(root, x, y);
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}), TensorB(out));
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);

  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

tensorflow::FunctionDef AddFunctionDef() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      absl::StrCat("    signature {"
                   "      name: 'AddFunction'"
                   "      input_arg {"
                   "        name: 'a'"
                   "        type: DT_FLOAT"
                   "      }"
                   "      input_arg {"
                   "        name: 'b'"
                   "        type: DT_FLOAT"
                   "      }"
                   "      output_arg {"
                   "        name: 'sum'"
                   "        type: DT_FLOAT"
                   "      }"
                   "    }"
                   "    node_def {"
                   "      name: 'add'"
                   "      op: 'AddV2'"
                   "      input: 'a'"
                   "      input: 'b'",
                   "      attr {"
                   "        key: 'T'"
                   "        value {"
                   "          type: DT_FLOAT"
                   "        }"
                   "      }"
                   "    }"
                   "    ret {"
                   "      key: 'sum'"
                   "      value: 'add:z:0'"
                   "    }"),
      &def));
  return def;
}

TEST_F(EagerComputationTest, CallAddGraphDefWithFunctionDef) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  auto add_function_def = AddFunctionDef();
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);

  tensorflow::OutputList placeholders;
  placeholders.push_back(x);
  placeholders.push_back(y);

  std::vector<tensorflow::DataType> tout_list;
  for (const auto& output : add_function_def.signature().output_arg()) {
    tout_list.push_back(output.type());
  }

  tensorflow::NameAttrList f_attr;
  f_attr.set_name(add_function_def.signature().name());

  tensorflow::ops::StatefulPartitionedCall call_op(root, placeholders,
                                                   tout_list, f_attr);
  tensorflow::ops::Identity identity(root, call_op.operation.output(0));
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}),
                         TensorB(identity), std::nullopt, {add_function_def});
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);

  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, InvalidComputationProto) {
  federated_language::Computation comp_pb;
  federated_language::TensorFlow* tensorflow_pb = comp_pb.mutable_tensorflow();
  tensorflow::TensorProto tensor_pb;

  tensorflow_pb->mutable_graph_def()->PackFrom(tensor_pb);

  ASSERT_THAT(EagerComputation::FromProto(comp_pb.tensorflow()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}
}  // namespace

}  // namespace tensorflow_federated
