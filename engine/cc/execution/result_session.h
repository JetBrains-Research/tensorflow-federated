#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "engine/cc/execution/plan.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

class ResultSessionNative {
 public:
  ResultSessionNative(
    tensorflow::GraphDef&& graph,
    const engine::tff::TensorflowSpec& spec,
    const engine::tff::ServerResultIORouter& router
  );

  absl::Status Run(
    const std::string& intermediate_ckpt_path,
    const std::string& aggregated_ckpt_path,
    const std::string& server_ckpt_path
  );

  static absl::StatusOr<std::unique_ptr<ResultSessionNative>> Create(
    tensorflow::GraphDef&& graph,
    const engine::tff::TensorflowSpec& spec,
    const engine::tff::ServerResultIORouter& router
  );

 private:
  tensorflow::GraphDef graph_;
  engine::tff::TensorflowSpec spec_;
  engine::tff::ServerResultIORouter router_;
};
