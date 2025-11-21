package org.jetbrains.tff.engine;

import static com.google.common.base.Preconditions.checkArgument;

public final class PrepareSession implements AutoCloseable {
  public PrepareSession(long handle) {
    checkArgument(handle != 0);
    this.sessionHandle = new NativeHandle(handle);
  }

  public void run(
    String serverCheckpointPath,
    String clientCheckpointPath,
    String intermediateCheckpointPath
  ) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      runPrepare(scopedHandle.get(), serverCheckpointPath, clientCheckpointPath, intermediateCheckpointPath);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  @Override
  protected void finalize() throws Throwable {
    assert !sessionHandle.isValid();
  }

  private final NativeHandle sessionHandle;
  private static IllegalStateException onExecutionException(ExecutionException e) {
    return new IllegalStateException("Native prepare session exception", e);
  }

  @Override
  public void close() {
    if (!sessionHandle.isValid()) {
      return;
    }
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      closeNative(scopedHandle.release());
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  static {
    NativeLibraryLoader.getInstance().loadLibrary();
  }

  /// CAREFUL: don't make the following native calls static because it can cause a race condition
  /// between the native execution and the object finalize() call.
  native void closeNative(long session) throws ExecutionException;
  native void runPrepare(
    long session,
    String serverCheckpointPath,
    String clientCheckpointPath,
    String intermediateCheckpointPath
  ) throws ExecutionException;
}
