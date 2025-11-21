
load("@rules_java//java:java_binary.bzl", "java_binary")

java_binary(
	name = "jni_tool",
	srcs = ["JniTool.java"],
	main_class = "org.jetbrains.tff.engine.JniTool",
	deps = [
		"//engine/java/src/main/org/jetbrains/tff/engine",
	],
)

