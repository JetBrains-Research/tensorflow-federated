// Tool for creating and parsing plans

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "absl/flags/usage.h"
#include "absl/flags/flag.h"
#include "engine/cc/execution/plan.pb.h"
#include "google/protobuf/util/json_util.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

ABSL_FLAG(std::string, server, "", "Server checkpoint path");
ABSL_FLAG(std::string, client, "", "Client checkpoint path");
ABSL_FLAG(std::string, inter, "", "Intermediate checkpoint path");
ABSL_FLAG(std::string, aggr, "", "Aggregated checkpoint path");

constexpr const char* kUsageMessage = R"(Usage:
  plan_tool build <output_bin>
    Build a binary Plan proto from JSON read from stdin.
  plan_tool parse <input_bin>
    Parse a binary Plan proto to JSON written to stdout.
  plan_tool sample
    Print a minimal Plan JSON with only ServerPhaseV2.aggregations filled.
  plan_tool prepare <plan.bin> --server=<ckpt_path> --client=<ckpt_path> --inter=<ckpt_path>
    Prepare checkpoints for testing.
)";

absl::StatusOr<engine::tff::Plan> parse(const std::string& plan_path) {
  std::ifstream in(plan_path, std::ios::binary);
  if (!in) {
    return absl::InvalidArgumentError("Failed to open input binary: " + plan_path);
  }
  engine::tff::Plan plan;
  if (!plan.ParseFromIstream(&in)) {
    return absl::InvalidArgumentError("Failed to parse Plan proto from binary");
  }

  return plan;
}

absl::Status prepare(const std::string& plan_path) {
  auto plan = parse(plan_path);
  if (!plan.ok()) {
    return plan.status();
  }

  const auto server_ckpt = absl::GetFlag(FLAGS_server);
  const auto client_ckpt = absl::GetFlag(FLAGS_client);
  const auto inter_ckpt = absl::GetFlag(FLAGS_inter);
  if (server_ckpt.empty() || client_ckpt.empty() || inter_ckpt.empty()) {
    return absl::InvalidArgumentError("Checkpoint paths are required.");
  }

  if (!plan->has_server_graph_prepare_bytes()) {
    return absl::InvalidArgumentError("Plan does not contain server_graph_prepare_bytes");
  }

  const auto& any_bytes = plan->server_graph_prepare_bytes();
  tensorflow::GraphDef graph_def;
  if (!any_bytes.UnpackTo(&graph_def)) {
    return absl::InvalidArgumentError("Failed to unpack GraphDef from server_graph_prepare_bytes");
  }

  tensorflow::SessionOptions session_options;
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
  tensorflow::Status status = session->Create(graph_def);
  if (!status.ok()) {
    return absl::InvalidArgumentError("Failed to create graph: " + status.ToString());
  }

  const auto& router = plan->phase(0).server_phase_v2().prepare_router();
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
  tensorflow::Tensor server_state_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  server_state_tensor.scalar<tensorflow::tstring>()() = server_ckpt;
  feed_dict.push_back({router.prepare_server_state_input_filepath_tensor_name(), server_state_tensor});

  tensorflow::Tensor client_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  client_tensor.scalar<tensorflow::tstring>()() = client_ckpt;
  feed_dict.push_back({router.prepare_output_filepath_tensor_name(), client_tensor});

  tensorflow::Tensor intermediate_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  intermediate_tensor.scalar<tensorflow::tstring>()() = inter_ckpt;
  feed_dict.push_back({router.prepare_intermediate_state_output_filepath_tensor_name(), intermediate_tensor});

  const auto& tensorflow_spec = plan->phase(0).server_phase_v2().tensorflow_spec_prepare();
  std::vector<std::string> target_node_names;
  for (const auto& target_node : tensorflow_spec.target_node_names()) {
    target_node_names.push_back(target_node);
  }

  std::vector<tensorflow::Tensor> outputs;
  const auto run_status = session->Run(feed_dict, {}, target_node_names, &outputs);
  if (!run_status.ok()) {
    return absl::InvalidArgumentError("Failed to run session: " + run_status.ToString());
  }

  std::cout << "Result phase finished successfully." << std::endl;
  return absl::OkStatus();
}

absl::Status result(const std::string& plan_path) {
  auto plan = parse(plan_path);
  if (!plan.ok()) {
    return plan.status();
  }

  const auto server_ckpt = absl::GetFlag(FLAGS_server);
  const auto aggr_ckpt = absl::GetFlag(FLAGS_aggr);
  const auto inter_ckpt = absl::GetFlag(FLAGS_inter);
  if (server_ckpt.empty() || aggr_ckpt.empty() || inter_ckpt.empty()) {
    return absl::InvalidArgumentError("Checkpoint paths are required.");
  }

  if (!plan->has_server_graph_result_bytes()) {
    return absl::InvalidArgumentError("Plan does not contain server_graph_result_bytes");
  }

  const auto& any_bytes = plan->server_graph_result_bytes();
  tensorflow::GraphDef graph_def;
  if (!any_bytes.UnpackTo(&graph_def)) {
    return absl::InvalidArgumentError("Failed to unpack GraphDef from server_graph_result_bytes");
  }

  tensorflow::SessionOptions session_options;
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
  tensorflow::Status status = session->Create(graph_def);
  if (!status.ok()) {
    return absl::InvalidArgumentError("Failed to create graph: " + status.ToString());
  }

  const auto& router = plan->phase(0).server_phase_v2().result_router();
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;
  tensorflow::Tensor server_state_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  server_state_tensor.scalar<tensorflow::tstring>()() = server_ckpt;
  feed_dict.push_back({router.result_server_state_output_filepath_tensor_name(), server_state_tensor});

  tensorflow::Tensor aggr_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  aggr_tensor.scalar<tensorflow::tstring>()() = aggr_ckpt;
  feed_dict.push_back({router.result_aggregate_result_input_filepath_tensor_name(), aggr_tensor});

  tensorflow::Tensor intermediate_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  intermediate_tensor.scalar<tensorflow::tstring>()() = inter_ckpt;
  feed_dict.push_back({router.result_intermediate_state_input_filepath_tensor_name(), intermediate_tensor});

  const auto& tensorflow_spec = plan->phase(0).server_phase_v2().tensorflow_spec_result();
  std::vector<std::string> target_node_names;
  for (const auto& target_node : tensorflow_spec.target_node_names()) {
    target_node_names.push_back(target_node);
  }

  std::vector<tensorflow::Tensor> outputs;
  const auto run_status = session->Run(feed_dict, {}, target_node_names, &outputs);
  if (!run_status.ok()) {
    return absl::InvalidArgumentError("Failed to run session: " + run_status.ToString());
  }

  std::cout << "Result phase finished successfully." << std::endl;
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(kUsageMessage);
  absl::ParseCommandLine(argc, argv);
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() < 1) {
    std::cerr << kUsageMessage;
    return 1;
  }

  const std::string& command = args[0];
  if (command == "sample") {
    // Minimal Plan with only ServerPhaseV2.aggregations filled
    engine::tff::Plan plan;
    auto* spv2 = plan.add_phase()->mutable_server_phase_v2();
    auto* agg = spv2->add_aggregations();
    agg->set_intrinsic_uri("federated_sum");
    // Add a minimal input_tensor argument
    auto* arg = agg->add_intrinsic_args();
    auto* input_tensor = arg->mutable_input_tensor();
    input_tensor->set_name("client_tensor");
    input_tensor->set_dtype(::tensorflow::DT_FLOAT);
    input_tensor->mutable_shape()->add_dim()->set_size(10);
    // Add a minimal output_tensor
    auto* output_tensor = agg->add_output_tensors();
    output_tensor->set_name("aggregated_tensor");
    output_tensor->set_dtype(::tensorflow::DT_FLOAT);
    output_tensor->mutable_shape()->add_dim()->set_size(10);
    std::string json;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    auto status = google::protobuf::util::MessageToJsonString(plan, &json, options);
    if (!status.ok()) {
      std::cerr << "Failed to convert Plan proto to JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::cout << json;
    return 0;
  }
  if (command == "build") {
    if (args.size() != 2) {
      std::cerr << kUsageMessage;
      return 1;
    }
    const std::string& output_bin = args[1];
    std::string json((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    engine::tff::Plan plan;
    auto status = google::protobuf::util::JsonStringToMessage(json, &plan);
    if (!status.ok()) {
      std::cerr << "Failed to parse JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::ofstream out(output_bin, std::ios::binary);
    if (!out) {
      std::cerr << "Failed to open output file: " << output_bin << std::endl;
      return 1;
    }
    if (!plan.SerializeToOstream(&out)) {
      std::cerr << "Failed to serialize Plan proto." << std::endl;
      return 1;
    }
    std::cout << "Plan binary written to: " << output_bin << std::endl;
  } else if (command == "parse") {
    if (args.size() != 2) {
      std::cerr << kUsageMessage;
      return 1;
    }

    const auto plan = parse(args[1]);
    if (!plan.ok()) {
      std::cerr << "Failed to parse Plan proto: " << plan.status().message() << std::endl;
      return 1;
    }

    std::string json;
    google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = true;
    auto status = google::protobuf::util::MessageToJsonString(plan.value(), &json, options);
    if (!status.ok()) {
      std::cerr << "Failed to convert Plan proto to JSON: " << status.ToString() << std::endl;
      return 1;
    }
    std::cout << json;
  } else if(command == "prepare") {
    if (args.size() < 2) {
      std::cerr << "Usage: plan_tool prepare <plan.bin> --server=<ckpt_path> --client=<ckpt_path> --inter=<ckpt_path>" << std::endl;
      return 1;
    }

    auto status = prepare(args[1]);
    if (!status.ok()) {
      std::cerr << "Failed to prepare: " << status.message() << std::endl;
      return 1;
    }
  } else if(command == "result") {
    if (args.size() < 2) {
      std::cerr << "Usage: plan_tool result <plan.bin> --server=<ckpt_path> --client=<ckpt_path> --inter=<ckpt_path>" << std::endl;
      return 1;
    }

    auto status = result(args[1]);
    if (!status.ok()) {
      std::cerr << "Failed to result: " << status.message() << std::endl;
      return 1;
    }
  } else {
    std::cerr << kUsageMessage;
    return 1;
  }
  return 0;
}
