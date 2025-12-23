// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package org.jetbrains.tff.engine;

import static com.google.common.base.Preconditions.checkArgument;

public final class AggregationSession implements AutoCloseable {
  public void accumulate(String[] checkpointPaths) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      runAccumulate(scopedHandle.get(), checkpointPaths);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  public void mergeWith(byte[][] serialized) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      mergeWith(scopedHandle.get(), configuration, serialized);
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  public byte[] serialize() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return serialize(scopedHandle.get());
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  public byte[] report() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return runReport(scopedHandle.get());
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  @Override
  protected void finalize() throws Throwable {
    assert !sessionHandle.isValid();
  }

  private final NativeHandle sessionHandle;
  private final byte[] configuration;

  public AggregationSession(long handle, byte[] configuration) {
    checkArgument(handle != 0);
    this.sessionHandle = new NativeHandle(handle);
    this.configuration = configuration;
  }

  private static IllegalStateException onExecutionException(ExecutionException e) {
    return new IllegalStateException("Native aggregation session exception", e);
  }

  @Override
  public void close() {
    if (!sessionHandle.isValid()) {
      return;
    }
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      closeAggregationSession(scopedHandle.release());
    } catch (ExecutionException e) {
      throw onExecutionException(e);
    }
  }

  static {
    NativeLibraryLoader.getInstance().loadLibrary();
  }

  // Native API
  // ==========
  /// CAREFUL: don't make the following native calls static because it can cause a race condition
  /// between the native execution and the object finalize() call.

  native void closeAggregationSession(long session) throws ExecutionException;
  native void runAccumulate(long session, String[] checkpointPaths) throws ExecutionException;
  native void mergeWith(long session, byte[] configuration, byte[][] serialized)
      throws ExecutionException;

  native byte[] serialize(long session) throws ExecutionException;
  native byte[] runReport(long session) throws ExecutionException;
}
