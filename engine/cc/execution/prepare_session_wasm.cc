// WASM-specific implementation of PrepareSessionNative
// This provides stubs that can be filled in with WASM-compatible logic

#include "prepare_session.h"

PrepareSessionNative::PrepareSessionNative(
  tensorflow::GraphDef&& graph,
  const engine::tff::TensorflowSpec& spec,
  const engine::tff::ServerPrepareIORouter& router
) : graph_(std::move(graph)), spec_(spec), router_(router)
{
}

absl::Status PrepareSessionNative::Run(
  const std::string& server_ckpt_path,
  const std::string& client_ckpt_path,
  const std::string& intermediate_ckpt_path
) {
  // WASM stub implementation
  // In a real implementation, you would:
  // 1. Serialize the graph and checkpoint paths
  // 2. Send them to a server-side TensorFlow instance
  // 3. Receive the results back
  // 4. Return the appropriate status

  return absl::UnimplementedError(
    "TensorFlow execution in WASM requires a server-side implementation. "
    "Graph serialization and remote execution not yet implemented."
  );
}

absl::StatusOr<std::unique_ptr<PrepareSessionNative>> PrepareSessionNative::Create(
  tensorflow::GraphDef&& graph,
  const engine::tff::TensorflowSpec& spec,
  const engine::tff::ServerPrepareIORouter& router
) {
  return std::make_unique<PrepareSessionNative>(
    std::move(graph), spec, router
  );
}
