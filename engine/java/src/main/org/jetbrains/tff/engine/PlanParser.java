package org.jetbrains.tff.engine;

public class PlanParser {
  static {
    NativeLibraryLoader.getInstance().loadLibrary();
  }

  private final byte[] plan;

  public PlanParser(byte[] plan) {
    this.plan = plan;
  }

  public AggregationSession createAggregationSession() {
    try {
      byte[] configuration = extractConfiguration(plan);
      long sessionHandle = createAggregationSessionHandle(configuration);
      return new AggregationSession(sessionHandle, configuration);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  public PrepareSession createPrepareSession() {
    try {
      long sessionHandle = createPrepareSessionHandle(plan);
      return new PrepareSession(sessionHandle);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  public ResultSession createResultSession() {
    try {
      long sessionHandle = createResultSessionHandle(plan);
      return new ResultSession(sessionHandle);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

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

  native long createAggregationSessionHandle(byte[] plan) throws ExecutionException;
  native long createPrepareSessionHandle(byte[] plan) throws ExecutionException;
  native long createResultSessionHandle(byte[] plan) throws ExecutionException;
  native byte[] createClientPhase(byte[] plan, long iterationNumber) throws ExecutionException;
  native byte[] extractConfiguration(byte[] plan) throws ExecutionException;
}
