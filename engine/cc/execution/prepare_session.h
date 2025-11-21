#pragma once

#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "engine/cc/execution/plan.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

class PrepareSessionNative {
 public:
  PrepareSessionNative(
    tensorflow::GraphDef&& graph,
    const engine::tff::TensorflowSpec& spec,
    const engine::tff::ServerPrepareIORouter& router
  );

  absl::Status Run(
    const std::string& server_ckpt_path,
    const std::string& client_ckpt_path,
    const std::string& intermediate_ckpt_path
  );

  static absl::StatusOr<std::unique_ptr<PrepareSessionNative>> Create(
    tensorflow::GraphDef&& graph,
    const engine::tff::TensorflowSpec& spec,
    const engine::tff::ServerPrepareIORouter& router
  );

 private:
  tensorflow::GraphDef graph_;
  engine::tff::TensorflowSpec spec_;
  engine::tff::ServerPrepareIORouter router_;
};
