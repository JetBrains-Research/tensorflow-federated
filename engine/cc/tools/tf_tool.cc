#include <iostream>

#include "absl/flags/usage.h"
#include "absl/flags/parse.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

constexpr const char* kUsageMessage = R"(Usage:
  tf_tool exec <graph_bin> <input_tensor_bin> <output_tensor_bin>
    Run graph with input tensor and write output tensor to file.
)";

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(kUsageMessage);
  absl::ParseCommandLine(argc, argv);
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() < 1) {
    std::cerr << kUsageMessage;
    return 1;
  }

  const std::string& command = args[0];
  if (command == "exec") {
    tensorflow::SessionOptions session_options;
    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(session_options));
    (void)session;
  }
}
