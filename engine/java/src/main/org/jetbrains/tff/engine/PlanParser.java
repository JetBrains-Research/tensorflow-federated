package org.jetbrains.tff.engine;

public class PlanParser {
  static {
    NativeLibraryLoader.getInstance().loadLibrary();
  }

  private final byte[] plan;

  public PlanParser(byte[] plan) {
    this.plan = plan;
  }

  /// Creates a new session, based on the plan.
  public AggregationSession createAggregationSession() {
    try {
      byte[] configuration = extractConfiguration(plan);
      long sessionHandle = createAggregationSessionHandle(configuration);
      return new AggregationSession(sessionHandle, configuration);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  /// Builds the client phase message with respect of iteration number.
  public byte[] createClientPhase(long iterationNumber) {
    try {
      return createClientPhase(plan, iterationNumber);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  private static IllegalStateException onExecutionException(ExecutionException e) {
    return new IllegalStateException("Plan parser exception", e);
  }

  // Native API
  // ==========
  /// CAREFUL: don't make the following native calls static because it can cause a race condition
  /// between the native execution and the object finalize() call.

  /// Starts a session based on the given plan, returning a handle for it. */
  native long createAggregationSessionHandle(byte[] plan) throws ExecutionException;

  /// Builds the client phase message with respect of iteration number.
  native byte[] createClientPhase(byte[] plan, long iterationNumber) throws ExecutionException;

  /// Extracts the configuration from the plan.
  native byte[] extractConfiguration(byte[] plan) throws ExecutionException;
}
