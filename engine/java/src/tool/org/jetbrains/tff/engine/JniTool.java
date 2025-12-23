package org.jetbrains.tff.engine;

import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Arrays;

public class JniTool {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java JniTool <command> [args...]");
            System.exit(1);
        }

        String cmd = args[0];
        if (cmd.equals("aggregate")) {
            Aggregate(Arrays.copyOfRange(args, 1, args.length));
        } else if (cmd.equals("prepare")) {
            Prepare(Arrays.copyOfRange(args, 1, args.length));
        } else if (cmd.equals("result")) {
            Result(Arrays.copyOfRange(args, 1, args.length));
        } else {
            System.err.println("Unknown command: " + cmd);
            System.exit(1);
        }
    }

    private static void Aggregate(String[] args) throws Exception{
        if (args.length != 4) {
            System.err.println("Usage: java jni_tool aggregate <plan_path> <left_ckpt_path> <right_ckpt_path> <output_ckpt_path>");
            System.exit(1);
        }

        String planPath = args[0];
        String leftCkptPath = args[1];
        String rightCkptPath = args[2];
        String outputCkptPath = args[3];

        byte[] planBytes = Files.readAllBytes(Paths.get(planPath));
        String[] checkpointPaths = new String[2];
        checkpointPaths[0] = leftCkptPath;
        checkpointPaths[1] = rightCkptPath;

        var parser = new PlanParser(planBytes);
        AggregationSession session = parser.createAggregationSession();
        session.accumulate(checkpointPaths);
        byte[] aggregatedCheckpoint = session.report();

        try (FileOutputStream out = new FileOutputStream(outputCkptPath)) {
            out.write(aggregatedCheckpoint);
        }

        System.out.println("Aggregated checkpoint written to: " + outputCkptPath);
    }

    private static void Prepare(String[] args) throws Exception{
        if (args.length != 4) {
            System.err.println("Usage: java jni_tool prepare <plan_path> <server_ckpt_path> <client_ckpt_path> <inter_ckpt_path>");
            System.exit(1);
        }

        String planPath = args[0];
        String serverCkptPath = args[1];
        String clientCkptPath = args[2];
        String interCkptPath = args[3];

        byte[] planBytes = Files.readAllBytes(Paths.get(planPath));
        var parser = new PlanParser(planBytes);
        PrepareSession session = parser.createPrepareSession();
        session.run(serverCkptPath, clientCkptPath, interCkptPath);
        System.out.println("Intermediate checkpoint written to: " + interCkptPath);
        System.out.println("Client checkpoint [maybe] written to: " + clientCkptPath);
    }

    private static void Result(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: java jni_tool result <plan_path> <inter_ckpt_path> <aggr_ckpt_path> <server_ckpt_path>");
            System.exit(1);
        }

        String planPath = args[0];
        String interCkptPath = args[1];
        String aggrCkptPath = args[2];
        String serverCkptPath = args[3];

        byte[] planBytes = Files.readAllBytes(Paths.get(planPath));
        var parser = new PlanParser(planBytes);
        ResultSession session = parser.createResultSession();
        session.run(interCkptPath, aggrCkptPath, serverCkptPath);
        System.out.println("Server checkpoint written to: " + serverCkptPath);
    }
}
