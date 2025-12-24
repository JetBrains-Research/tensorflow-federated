/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jni.h>
#include "absl/status/status.h"
#include "util.h"
#include "prepare_session.h"
#include "result_session.h"
#include "engine/cc/execution/plan.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_builder_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/tensorflow_checkpoint_parser_factory.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/platform.h"

#define JFUN(CLASS_NAME, METHOD_NAME) \
  Java_org_jetbrains_tff_engine_##CLASS_NAME##_##METHOD_NAME

constexpr const char* EXE_EXCEPTION_CLASS = "org/jetbrains/tff/engine/ExecutionException";

// Helper methods
// ==============
namespace {

using namespace tensorflow_federated::aggregation;
using IntrinsicArg = tensorflow_federated::aggregation::Configuration_IntrinsicConfig_IntrinsicArg;
using IntrinsicConfig = tensorflow_federated::aggregation::Configuration_IntrinsicConfig;
namespace agg = tensorflow_federated::aggregation::tensorflow;

// Throws a ExecutionException with the given status code and message in the
// JNI environment.
void ThrowExecutionException(JNIEnv* env, int code, const std::string& message) {
  jni::ThrowCustomStatusCodeException(env, EXE_EXCEPTION_CLASS, code, message);
}

void ThrowExecutionException(JNIEnv* env, const absl::Status& error) {
  ThrowExecutionException(env, (int)error.code(), std::string(error.message()));
}

std::string Message(const absl::Status& status) {
  return std::string(status.message());
}

absl::StatusOr<CheckpointAggregator*> AsAggregator(jlong handle) {
  if (handle == 0) {
    return absl::InvalidArgumentError("Invalid session handle (session closed?)");
  }

  return reinterpret_cast<CheckpointAggregator*>(handle);
}

absl::StatusOr<PrepareSessionNative*> AsPrepareSessionNative(jlong handle) {
  if (handle == 0) {
    return absl::InvalidArgumentError("Invalid session handle (session closed?)");
  }

  return reinterpret_cast<PrepareSessionNative*>(handle);
}

absl::StatusOr<ResultSessionNative*> AsResultSessionNative(jlong handle) {
  if (handle == 0) {
    return absl::InvalidArgumentError("Invalid session handle (session closed?)");
  }

  return reinterpret_cast<ResultSessionNative*>(handle);
}

absl::StatusOr<IntrinsicArg> ConvertIntrinsicArg(const engine::tff::ServerAggregationConfig_IntrinsicArg& arg) {
  if (arg.has_state_tensor()) {
    return absl::InvalidArgumentError("State tensors are not supported yet.");
  }

  IntrinsicArg result;
  const auto input_tensor = agg::ToAggTensorSpec(arg.input_tensor());
  if (!input_tensor.ok()) {
    return absl::InvalidArgumentError("Failed to convert input tensor spec: " + Message(input_tensor.status()));
  }
  result.mutable_input_tensor()->CopyFrom(input_tensor->ToProto());
  return result;
}

absl::StatusOr<IntrinsicConfig> ConvertConfig(const engine::tff::ServerAggregationConfig& config) {
  IntrinsicConfig result;
  for (const auto& aggregation : config.inner_aggregations()) {
    const auto converted = ConvertConfig(aggregation);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert inner aggregation config: " + Message(converted.status()));
    }

    *result.add_inner_intrinsics() = converted.value();
  }

  for (const auto& output : config.output_tensors()) {
    const auto converted = agg::ToAggTensorSpec(output);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert output tensor spec: " + Message(converted.status()));
    }

    *result.add_output_tensors() = converted->ToProto();
  }

  for (const auto& intrinsic_arg : config.intrinsic_args()) {
    const auto converted = ConvertIntrinsicArg(intrinsic_arg);
    if (!converted.ok()) {
      return absl::InvalidArgumentError("Failed to convert intrinsic arg: " + Message(converted.status()));
    }

    *result.add_intrinsic_args() = converted.value();
  }

  result.set_intrinsic_uri(config.intrinsic_uri());
  return result;
}

absl::StatusOr<Configuration>
ExtractAggregationConfigurationFromPlan(const engine::tff::Plan& plan) {
  if (plan.phase_size() == 0) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "No phases in the plan.");
  }

  if (!plan.phase(0).has_server_phase_v2()) {
    return absl::Status(absl::StatusCode::kInvalidArgument, "No server phases in the plan.");
  }

  Configuration result;
  const auto& server_phase_v2 = plan.phase(0).server_phase_v2();
  for (const auto& config : server_phase_v2.aggregations()) {
    const auto converted = ConvertConfig(config);
    if (!converted.ok()) {
      return absl::Status(converted.status().code(), "Failed to convert aggregation config: " + Message(converted.status()));
    }

    *result.add_intrinsic_configs() = *converted;
  }

  return result;
}

absl::StatusOr<engine::tff::ClientOnlyPlan>
ExtractClientOnlyPlan(const engine::tff::Plan& plan) {
  engine::tff::ClientOnlyPlan result;
  result.mutable_phase()->CopyFrom(plan.phase(0).client_phase());
  result.set_graph(plan.client_graph_bytes().value());
  result.set_tflite_graph(plan.client_tflite_graph_bytes());
  if (plan.has_tensorflow_config_proto()) {
    result.mutable_tensorflow_config_proto()->CopyFrom(plan.tensorflow_config_proto());
  }

  return result;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL JFUN(PlanParser, createAggregationSessionHandle)(
    JNIEnv* env, jobject, jbyteArray configurationByteArray) {
  auto config = jni::ParseProtoFromJByteArray<Configuration>(env, configurationByteArray);
  if (!config.ok()) {
    ThrowExecutionException(env, config.status());
    return 0;
  }

  auto result = CheckpointAggregator::Create(*config);
  if (!result.ok()) {
    ThrowExecutionException(env, result.status());
    return 0;
  }

  return reinterpret_cast<jlong>(result->release());
}

extern "C" JNIEXPORT jlong JNICALL JFUN(PlanParser, createPrepareSessionHandle)(
  JNIEnv* env, jobject, jbyteArray planByteArray) {
  auto plan = jni::ParseProtoFromJByteArray<engine::tff::Plan>(env, planByteArray);
  if (!plan.ok()) {
    ThrowExecutionException(env, plan.status());
    return {};
  }

  if (!plan->has_server_graph_prepare_bytes()) {
    ThrowExecutionException(env, absl::InvalidArgumentError("Plan does not contain server_graph_prepare_bytes"));
    return {};
  }

  ::tensorflow::GraphDef graph;
  if (!plan->server_graph_prepare_bytes().UnpackTo(&graph)) {
    ThrowExecutionException(env, absl::InvalidArgumentError("Failed to unpack GraphDef from server_graph_prepare_bytes"));
    return {};
  }

  const auto router = plan->phase(0).server_phase_v2().prepare_router();
  const auto spec = plan->phase(0).server_phase_v2().tensorflow_spec_prepare();
  auto session = PrepareSessionNative::Create(std::move(graph), spec, router);
  return reinterpret_cast<jlong>(session->release());
}

extern "C" JNIEXPORT jlong JNICALL JFUN(PlanParser, createResultSessionHandle)(
  JNIEnv* env, jobject, jbyteArray planByteArray) {
  auto plan = jni::ParseProtoFromJByteArray<engine::tff::Plan>(env, planByteArray);
  if (!plan.ok()) {
    ThrowExecutionException(env, plan.status());
    return {};
  }

  if (!plan->has_server_graph_result_bytes()) {
    ThrowExecutionException(env, absl::InvalidArgumentError("Plan does not contain server_graph_result_bytes"));
    return {};
  }

  ::tensorflow::GraphDef graph;
  if (!plan->server_graph_result_bytes().UnpackTo(&graph)) {
    ThrowExecutionException(env, absl::InvalidArgumentError("Failed to unpack GraphDef from server_graph_prepare_bytes"));
    return {};
  }

  const auto router = plan->phase(0).server_phase_v2().result_router();
  const auto spec = plan->phase(0).server_phase_v2().tensorflow_spec_result();
  auto session = ResultSessionNative::Create(std::move(graph), spec, router);
  return reinterpret_cast<jlong>(session->release());
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(PlanParser, extractConfiguration)(
  JNIEnv* env,
  jobject,
  jbyteArray planByteArray
) {
  const auto plan = jni::ParseProtoFromJByteArray<engine::tff::Plan>(env, planByteArray);
  if (!plan.ok()) {
    ThrowExecutionException(env, plan.status());
    return {};
  }

  const auto config = ExtractAggregationConfigurationFromPlan(*plan);
  if (!config.ok()) {
    ThrowExecutionException(env, config.status());
    return {};
  }

  const auto result = jni::SerializeProtoToJByteArray(env, *config);
  if (!result.ok()) {
    ThrowExecutionException(env, result.status());
    return {};
  }

  return *result;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(PlanParser, createClientPhase)(
  JNIEnv* env,
  jobject,
  jbyteArray planByteArray,
  jlong iterationNumber
) {
  const auto plan = jni::ParseProtoFromJByteArray<engine::tff::Plan>(env, planByteArray);
  if (!plan.ok()) {
    ThrowExecutionException(env, plan.status());
    return {};
  }

  auto client_only_plan = ExtractClientOnlyPlan(*plan);
  if (!client_only_plan.ok()) {
    ThrowExecutionException(env, client_only_plan.status());
    return {};
  }

  if (iterationNumber >= 0) {
    client_only_plan->mutable_client_persisted_data()->set_min_sep_policy_index(static_cast<int64_t>(iterationNumber));
  }

  const auto result = jni::SerializeProtoToJByteArray(env, *client_only_plan);
  if (!result.ok()) {
    ThrowExecutionException(env, result.status());
    return {};
  }

  return *result;
}

extern "C" JNIEXPORT void JNICALL JFUN(AggregationSession, mergeWith)(
  JNIEnv* env,
  jobject,
  jlong handle,
  jbyteArray configurationByteArray,
  jobjectArray serializedStatePaths
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowExecutionException(env, aggregator.status());
    return;
  }

  auto config = jni::ParseProtoFromJByteArray<Configuration>(env, configurationByteArray);
  if (!config.ok()) {
    ThrowExecutionException(env, config.status());
    return;
  }

  int len = env->GetArrayLength(serializedStatePaths);
  for (int i = 0; i < len; i++) {
    jstring serializedStatePath = (jstring)env->GetObjectArrayElement(serializedStatePaths, i);
    if (jni::CheckJniException(env, "GetObjectArrayElement") != absl::OkStatus()) {
      ThrowExecutionException(env,  absl::InternalError("Failed to get array element"));
      return;
    }

    auto serializedStatePathStr = jni::JstringToString(env, serializedStatePath);
    if (!serializedStatePathStr.ok()) {
      ThrowExecutionException(env, serializedStatePathStr.status());
      return;
    }

    auto serializedStateContent = tensorflow_federated::ReadFileToCord(serializedStatePathStr.value());
    if (!serializedStateContent.ok()) {
      ThrowExecutionException(env, serializedStateContent.status());
      return;
    }

    std::string serializedStateStr;
    absl::CopyCordToString(serializedStateContent.value(), &serializedStateStr);

    auto other_aggregator = CheckpointAggregator::Deserialize(config.value(), serializedStateStr);

    if (!other_aggregator.ok()) {
      ThrowExecutionException(env, other_aggregator.status());
      return;
    }

    if (auto status = aggregator.value()->MergeWith(std::move(*(other_aggregator.value()))); !status.ok()) {
      ThrowExecutionException(env, status);
      return;
    }
  }
  return;
}

extern "C" JNIEXPORT void JNICALL JFUN(AggregationSession, closeNative)(JNIEnv* env, jobject obj, jlong handle) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowExecutionException(env, aggregator.status());
    return;
  }

  delete aggregator.value();
}

extern "C" JNIEXPORT void JNICALL JFUN(AggregationSession, runAccumulate)(
  JNIEnv* env,
  jobject,
  jlong handle,
  jobjectArray checkpointPaths
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowExecutionException(env, aggregator.status());
    return;
  }

  const auto len = env->GetArrayLength(checkpointPaths);
  if (auto status = jni::CheckJniException(env, "Failed to get array length"); !status.ok()) {
    ThrowExecutionException(env, status);
    return;
  }

  for (int i = 0; i < len; i++) {
    jstring checkpointPath = (jstring)env->GetObjectArrayElement(checkpointPaths, i);
    if (auto status = jni::CheckJniException(env, "GetObjectArrayElement"); !status.ok()) {
      ThrowExecutionException(env, status);
      return;
    }

    auto checkpointPathStr = jni::JstringToString(env, checkpointPath);
    if (!checkpointPathStr.ok()) {
      ThrowExecutionException(env, checkpointPathStr.status());
      return;
    }

    auto checkpointContent = tensorflow_federated::ReadFileToCord(checkpointPathStr.value());
    if (!checkpointContent.ok()) {
      ThrowExecutionException(env, checkpointContent.status());
      return;
    }

    agg::TensorflowCheckpointParserFactory parser_factory;
    auto parser = parser_factory.Create(checkpointContent.value());
    if (!parser.ok()) {
      ThrowExecutionException(env, parser.status());
      return;
    }

    if (auto status = aggregator.value()->Accumulate(*(parser.value())); !status.ok()) {
      ThrowExecutionException(env, status);
      return;
    }
  }
  return;
}

extern "C" JNIEXPORT jstring JNICALL JFUN(AggregationSession, runReport)(
  JNIEnv* env,
  jobject,
  jlong handle,
  jstring outputPath
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowExecutionException(env, aggregator.status());
    return {};
  }

  auto outputPathStr = jni::JstringToString(env, outputPath);
  if (!outputPathStr.ok()) {
    ThrowExecutionException(env, outputPathStr.status());
    return {};
  }

  agg::TensorflowCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  absl::Status status = aggregator.value()->Report(*builder);
  if (!status.ok()) {
    ThrowExecutionException(env, status);
    return {};
  }

  auto res = builder->Build();
  if (!res.ok()) {
    ThrowExecutionException(env, res.status());
    return {};
  }

  auto writeStatus = tensorflow_federated::WriteCordToFile(outputPathStr.value(), *res);
  if (!writeStatus.ok()) {
    ThrowExecutionException(env, writeStatus);
    return {};
  }

  return outputPath;
}

extern "C" JNIEXPORT jbyteArray JNICALL JFUN(AggregationSession, serialize)(
  JNIEnv* env,
  jobject,
  jlong handle
) {
  auto aggregator = AsAggregator(handle);
  if (!aggregator.ok()) {
    ThrowExecutionException(env, aggregator.status());
    return {};
  }

  auto serialized = std::move(*aggregator.value()).Serialize();
  if (!serialized.ok()) {
    ThrowExecutionException(env, serialized.status());
    return {};
  }

  const auto serializedAggregator = serialized.value();
  auto byteArray = env->NewByteArray(serializedAggregator.length());
  if (auto status = jni::CheckJniException(env, "NewByteArray"); !status.ok()) {
    ThrowExecutionException(env, status);
    return {};
  }

  const auto aggregatorBytes = reinterpret_cast<const jbyte*>(serializedAggregator.c_str());
  env->SetByteArrayRegion(byteArray, 0, serializedAggregator.length(), aggregatorBytes);
  if (auto status = jni::CheckJniException(env, "SetByteArrayRegion"); !status.ok()) {
    ThrowExecutionException(env, status);
    return {};
  }

  return byteArray;
}

extern "C" JNIEXPORT void JNICALL JFUN(PrepareSession, runPrepare)(
  JNIEnv* env,
  jobject,
  jlong handle,
  jstring server_ckpt_path,
  jstring client_ckpt_path,
  jstring intermediate_ckpt_path
) {
  auto session = AsPrepareSessionNative(handle);
  if (!session.ok()) {
    ThrowExecutionException(env, session.status());
    return;
  }

  const auto server_ckpt_path_str = jni::JstringToString(env, server_ckpt_path);
  if (!server_ckpt_path_str.ok()) {
    ThrowExecutionException(env, server_ckpt_path_str.status());
    return;
  }

  const auto client_ckpt_path_str = jni::JstringToString(env, client_ckpt_path);
  if (!client_ckpt_path_str.ok()) {
    ThrowExecutionException(env, client_ckpt_path_str.status());
    return;
  }

  const auto intermediate_ckpt_path_str = jni::JstringToString(env, intermediate_ckpt_path);
  if (!intermediate_ckpt_path_str.ok()) {
    ThrowExecutionException(env, intermediate_ckpt_path_str.status());
    return;
  }

  const auto result = session.value()->Run(
    server_ckpt_path_str.value(),
    client_ckpt_path_str.value(),
    intermediate_ckpt_path_str.value()
  );

  if (!result.ok()) {
    ThrowExecutionException(env, result);
    return;
  }
}

extern "C" JNIEXPORT void JNICALL JFUN(PrepareSession, closeNative)(JNIEnv* env, jobject obj, jlong handle) {
  auto session = AsPrepareSessionNative(handle);
  if (!session.ok()) {
    ThrowExecutionException(env, session.status());
    return;
  }

  delete session.value();
}

extern "C" JNIEXPORT void JNICALL JFUN(ResultSession, runResult)(
  JNIEnv* env,
  jobject,
  jlong handle,
  jstring intermediate_ckpt_path,
  jstring aggregated_ckpt_path,
  jstring server_ckpt_path
) {
  auto session = AsResultSessionNative(handle);
  if (!session.ok()) {
    ThrowExecutionException(env, session.status());
    return;
  }

  const auto intermediate_ckpt_path_str = jni::JstringToString(env, intermediate_ckpt_path);
  if (!intermediate_ckpt_path_str.ok()) {
    ThrowExecutionException(env, intermediate_ckpt_path_str.status());
    return;
  }
  const auto aggregated_ckpt_path_str = jni::JstringToString(env, aggregated_ckpt_path);
  if (!aggregated_ckpt_path_str.ok()) {
    ThrowExecutionException(env, aggregated_ckpt_path_str.status());
    return;
  }

  const auto server_ckpt_path_str = jni::JstringToString(env, server_ckpt_path);
  if (!server_ckpt_path_str.ok()) {
    ThrowExecutionException(env, server_ckpt_path_str.status());
    return;
  }

  const auto result = session.value()->Run(intermediate_ckpt_path_str.value(), aggregated_ckpt_path_str.value(), server_ckpt_path_str.value());
  if (!result.ok()) {
    ThrowExecutionException(env, result);
    return;
  }
}

extern "C" JNIEXPORT void JNICALL JFUN(ResultSession, closeNative)(JNIEnv* env, jobject obj, jlong handle) {
  auto session = AsResultSessionNative(handle);
  if (!session.ok()) {
    ThrowExecutionException(env, session.status());
    return;
  }

  delete session.value();
}
