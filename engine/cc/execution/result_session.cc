#include "result_session.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

ResultSessionNative::ResultSessionNative(
  tensorflow::GraphDef&& graph,
  const engine::tff::TensorflowSpec& spec,
  const engine::tff::ServerResultIORouter& router
) : graph_(std::move(graph)), spec_(spec), router_(router)
{
}

absl::Status ResultSessionNative::Run(
  const std::string& intermediate_ckpt_path,
  const std::string& aggregated_ckpt_path,
  const std::string& server_ckpt_path
) {
  tensorflow::SessionOptions session_options;
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
  tensorflow::Status status = session->Create(graph_);
  if (!status.ok()) {
    return absl::InvalidArgumentError("Failed to create graph: " + status.ToString());
  }

  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
  tensorflow::Tensor server_state_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  server_state_tensor.scalar<tensorflow::tstring>()() = server_ckpt_path;
  feed_dict.push_back({router_.result_server_state_output_filepath_tensor_name(), server_state_tensor});

  tensorflow::Tensor aggr_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  aggr_tensor.scalar<tensorflow::tstring>()() = aggregated_ckpt_path;
  feed_dict.push_back({router_.result_aggregate_result_input_filepath_tensor_name(), aggr_tensor});

  tensorflow::Tensor intermediate_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  intermediate_tensor.scalar<tensorflow::tstring>()() = intermediate_ckpt_path;
  feed_dict.push_back({router_.result_intermediate_state_input_filepath_tensor_name(), intermediate_tensor});

  std::vector<std::string> target_node_names;
  for (const auto& target_node : spec_.target_node_names()) {
    target_node_names.push_back(target_node);
  }

  std::vector<tensorflow::Tensor> outputs;
  const auto run_status = session->Run(feed_dict, {}, target_node_names, &outputs);
  if (!run_status.ok()) {
    return absl::InvalidArgumentError("Failed to run session: " + run_status.ToString());
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ResultSessionNative>> ResultSessionNative::Create(
  tensorflow::GraphDef&& graph,
  const engine::tff::TensorflowSpec& spec,
  const engine::tff::ServerResultIORouter& router
) {
  return std::make_unique<ResultSessionNative>(std::move(graph), spec, router);
}
