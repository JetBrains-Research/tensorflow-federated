# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
package(
    default_visibility = ["//visibility:public"],
)

load("@rules_jvm_external//:defs.bzl", "java_export")
load("@rules_java//java:java_library.bzl", "java_library")

TFF_ENGINE_VERSION = "2.0.5"
TFF_ENGINE_PACKAGE = "org.jetbrains:tff-engine-{platform}:{version}"

java_library(
    name = "engine",
    srcs = [
        "AggregationSession.java",
        "NativeLibraryLoader.java",
        "ExecutionException.java",
        "NativeHandle.java",
        "PlanParser.java",
    ],
    resources = [":copy_execution_jni_to_resources"],
	resource_strip_prefix = "engine/java/src/main/org/jetbrains/tff/engine",
    deps = [
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

java_export(
    name = "engine_package_osx_x86_64",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(platform="osx-x86_64", version=TFF_ENGINE_VERSION),
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

java_export(
    name = "engine_package_osx_aarch64",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(platform="osx-aarch64", version=TFF_ENGINE_VERSION),
    target_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
)

java_export(
    name = "engine_package_linux_x86_64",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(platform="linux-x86_64", version=TFF_ENGINE_VERSION),
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

java_export(
    name = "engine_package_linux_aarch64",
    exports = [":engine"],
    maven_coordinates = TFF_ENGINE_PACKAGE.format(platform="linux-aarch64", version=TFF_ENGINE_VERSION),
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "macos_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "macos_aarch64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

alias(
    name = "publish",
    actual = select({
        ":macos_x86_64": ":engine_package_osx_x86_64.publish",
        ":macos_aarch64": ":engine_package_osx_aarch64.publish",
        ":linux_x86_64": ":engine_package_linux_x86_64.publish",
        ":linux_aarch64": ":engine_package_linux_aarch64.publish",
    }),
)

genrule(
    name = "copy_execution_jni_to_resources_dylib",
    srcs = ["//engine/cc/execution:execution-jni"],
    outs = ["libexecution-jni.dylib"],
    cmd = "cp $< $@",
    target_compatible_with = ["@platforms//os:macos"],
)

genrule(
    name = "copy_execution_jni_to_resources_so",
    srcs = ["//engine/cc/execution:execution-jni"],
    outs = ["libexecution-jni.so"],
    cmd = "cp $< $@",
    target_compatible_with = ["@platforms//os:linux"],
)

alias(
    name = "copy_execution_jni_to_resources",
    actual = select({
        "@platforms//os:macos": ":copy_execution_jni_to_resources_dylib",
        "@platforms//os:linux": ":copy_execution_jni_to_resources_so",
    }),
)
