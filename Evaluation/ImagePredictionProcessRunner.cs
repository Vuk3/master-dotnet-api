using System.Diagnostics;
using System.Text;
using System.Text.Json;
using dotnet_api.Responses;

namespace dotnet_api.Evaluation;

public static class ImagePredictionProcessRunner
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
    };

    public static PredictionBatchRunResult PredictBatch(
        IReadOnlyList<PredictionBatchItem> items,
        EvaluationOptions options,
        Action<string>? progress
    )
    {
        if (items.Count == 0)
        {
            return new PredictionBatchRunResult(
                new Dictionary<string, PredictionResponse>(StringComparer.OrdinalIgnoreCase),
                []
            );
        }

        Directory.CreateDirectory(options.OutputDirectory);

        var predictions = new Dictionary<string, PredictionResponse>(StringComparer.OrdinalIgnoreCase);
        var failures = new Dictionary<string, PredictionFailure>(StringComparer.OrdinalIgnoreCase);
        var batchSize = Math.Max(1, options.PredictionBatchSize);
        var timeoutSeconds = Math.Max(1, options.PredictionTimeoutSeconds);
        var retries = Math.Max(0, options.PredictionRetries);

        for (var offset = 0; offset < items.Count; offset += batchSize)
        {
            var batch = items.Skip(offset).Take(batchSize).ToList();
            var run = RunWorkerBatch(
                batch,
                options.ModelId,
                options.OutputDirectory,
                timeoutSeconds,
                progress
            );

            MergePredictions(run, predictions);

            foreach (var item in batch.Where(item => !predictions.ContainsKey(item.ImageId)))
            {
                var failure = GetFailureInfo(run, item);

                for (var retry = 1; retry <= retries && !predictions.ContainsKey(item.ImageId); retry++)
                {
                    progress?.Invoke(
                        $"Retrying {item.Index}/{item.Total}: {item.FileName} (attempt {retry + 1}/{retries + 1})"
                    );

                    var retryRun = RunWorkerBatch(
                        [item],
                        options.ModelId,
                        options.OutputDirectory,
                        timeoutSeconds,
                        progress
                    );
                    MergePredictions(retryRun, predictions);

                    if (!predictions.ContainsKey(item.ImageId))
                    {
                        failure = GetFailureInfo(retryRun, item);
                    }
                }

                if (!predictions.ContainsKey(item.ImageId))
                {
                    failures[item.ImageId] = new PredictionFailure(
                        item.ImageId,
                        item.FileName,
                        item.ImagePath,
                        failure.ExitCode,
                        Shorten(failure.Error)
                    );
                }
            }
        }

        return new PredictionBatchRunResult(predictions, failures.Values.ToList());
    }

    public static PredictionResponse PredictFile(
        string imagePath,
        string? modelId,
        string outputDirectory
    )
    {
        Directory.CreateDirectory(outputDirectory);
        var outputPath = Path.Combine(outputDirectory, $".prediction-{Guid.NewGuid():N}.json");

        try
        {
            using var process = StartWorker(imagePath, modelId, outputPath);
            var stdoutTask = process.StandardOutput.ReadToEndAsync();
            var stderrTask = process.StandardError.ReadToEndAsync();
            process.WaitForExit();

            var stdout = stdoutTask.GetAwaiter().GetResult();
            var stderr = stderrTask.GetAwaiter().GetResult();

            if (process.ExitCode != 0)
            {
                throw new PredictionProcessException(
                    process.ExitCode,
                    BuildErrorMessage(stdout, stderr, "Prediction worker failed.")
                );
            }

            if (!File.Exists(outputPath))
            {
                throw new PredictionProcessException(
                    process.ExitCode,
                    BuildErrorMessage(stdout, stderr, "Prediction worker did not write a result file.")
                );
            }

            var prediction = JsonSerializer.Deserialize<PredictionResponse>(
                File.ReadAllText(outputPath),
                JsonOptions
            );

            return prediction
                ?? throw new PredictionProcessException(
                    process.ExitCode,
                    "Prediction worker wrote an empty result."
                );
        }
        finally
        {
            TryDelete(outputPath);
        }
    }

    private static WorkerBatchResult RunWorkerBatch(
        IReadOnlyList<PredictionBatchItem> items,
        string? modelId,
        string outputDirectory,
        int timeoutSeconds,
        Action<string>? progress
    )
    {
        var batchId = Guid.NewGuid().ToString("N");
        var manifestPath = Path.Combine(outputDirectory, $".prediction-batch-{batchId}.manifest.json");
        var outputPath = Path.Combine(outputDirectory, $".prediction-batch-{batchId}.jsonl");

        try
        {
            File.WriteAllText(
                manifestPath,
                JsonSerializer.Serialize(new PredictionBatchManifest(modelId, items), JsonOptions)
            );

            using var process = StartBatchWorker(manifestPath, outputPath);
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();

            process.OutputDataReceived += (_, eventArgs) =>
            {
                if (string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    return;
                }

                stdout.AppendLine(eventArgs.Data);
                progress?.Invoke(eventArgs.Data);
            };
            process.ErrorDataReceived += (_, eventArgs) =>
            {
                if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                {
                    stderr.AppendLine(eventArgs.Data);
                }
            };

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            var exited = process.WaitForExit(timeoutSeconds * 1000);
            if (!exited)
            {
                TryKill(process);
                var timedOut = ReadBatchResults(outputPath, out var timeoutParseError);
                var timeoutMessage = $"Prediction worker timed out after {timeoutSeconds}s.";
                if (!string.IsNullOrWhiteSpace(timeoutParseError))
                {
                    timeoutMessage = $"{timeoutMessage} {timeoutParseError}";
                }

                return new WorkerBatchResult(
                    timedOut.Predictions,
                    timedOut.ItemErrors,
                    null,
                    timeoutMessage
                );
            }

            process.WaitForExit();

            var parsed = ReadBatchResults(outputPath, out var parseError);
            var processError = process.ExitCode == 0
                ? null
                : BuildErrorMessage(stdout.ToString(), stderr.ToString(), "Prediction batch worker failed.");

            if (!string.IsNullOrWhiteSpace(parseError))
            {
                processError = string.IsNullOrWhiteSpace(processError)
                    ? parseError
                    : $"{processError} | {parseError}";
            }

            return new WorkerBatchResult(
                parsed.Predictions,
                parsed.ItemErrors,
                process.ExitCode,
                processError
            );
        }
        finally
        {
            TryDelete(manifestPath);
            TryDelete(outputPath);
        }
    }

    private static Process StartBatchWorker(string manifestPath, string outputPath)
    {
        var executablePath = Environment.ProcessPath
            ?? throw new InvalidOperationException("Current process path could not be resolved.");
        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("predict-batch");
        startInfo.ArgumentList.Add("--manifest");
        startInfo.ArgumentList.Add(manifestPath);
        startInfo.ArgumentList.Add("--out");
        startInfo.ArgumentList.Add(outputPath);

        return Process.Start(startInfo)
            ?? throw new InvalidOperationException("Prediction batch worker could not be started.");
    }

    private static ParsedBatchResults ReadBatchResults(string outputPath, out string? parseError)
    {
        var predictions = new Dictionary<string, PredictionResponse>(StringComparer.OrdinalIgnoreCase);
        var itemErrors = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var parseErrors = new List<string>();

        if (!File.Exists(outputPath))
        {
            parseError = null;
            return new ParsedBatchResults(predictions, itemErrors);
        }

        var lineNumber = 0;
        foreach (var line in File.ReadLines(outputPath))
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            try
            {
                var result = JsonSerializer.Deserialize<PredictionBatchItemResult>(line, JsonOptions);
                if (result is null || string.IsNullOrWhiteSpace(result.ImageId))
                {
                    continue;
                }

                if (result.Prediction is not null)
                {
                    predictions[result.ImageId] = result.Prediction;
                }
                else if (!string.IsNullOrWhiteSpace(result.Error))
                {
                    itemErrors[result.ImageId] = result.Error;
                }
            }
            catch (Exception exception)
            {
                parseErrors.Add($"line {lineNumber}: {exception.Message}");
            }
        }

        parseError = parseErrors.Count == 0
            ? null
            : $"Could not parse {parseErrors.Count} worker result line(s): {string.Join("; ", parseErrors.Take(3))}";
        return new ParsedBatchResults(predictions, itemErrors);
    }

    private static void MergePredictions(
        WorkerBatchResult run,
        IDictionary<string, PredictionResponse> predictions
    )
    {
        foreach (var (imageId, prediction) in run.Predictions)
        {
            predictions[imageId] = prediction;
        }
    }

    private static FailureInfo GetFailureInfo(WorkerBatchResult run, PredictionBatchItem item)
    {
        if (run.ItemErrors.TryGetValue(item.ImageId, out var itemError))
        {
            return new FailureInfo(run.ExitCode, itemError);
        }

        if (!string.IsNullOrWhiteSpace(run.ProcessError))
        {
            return new FailureInfo(run.ExitCode, run.ProcessError);
        }

        return new FailureInfo(run.ExitCode, "Prediction worker did not return a result for this image.");
    }

    private static Process StartWorker(string imagePath, string? modelId, string outputPath)
    {
        var executablePath = Environment.ProcessPath
            ?? throw new InvalidOperationException("Current process path could not be resolved.");
        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        startInfo.ArgumentList.Add("predict-image");
        startInfo.ArgumentList.Add("--image");
        startInfo.ArgumentList.Add(imagePath);
        startInfo.ArgumentList.Add("--out");
        startInfo.ArgumentList.Add(outputPath);

        if (!string.IsNullOrWhiteSpace(modelId))
        {
            startInfo.ArgumentList.Add("--model");
            startInfo.ArgumentList.Add(modelId);
        }

        return Process.Start(startInfo)
            ?? throw new InvalidOperationException("Prediction worker could not be started.");
    }

    private static string BuildErrorMessage(string stdout, string stderr, string fallback)
    {
        var builder = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(stderr))
        {
            builder.Append(stderr.Trim());
        }

        if (!string.IsNullOrWhiteSpace(stdout))
        {
            if (builder.Length > 0)
            {
                builder.Append(" | ");
            }

            builder.Append(stdout.Trim());
        }

        return builder.Length == 0 ? fallback : builder.ToString();
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Best-effort cleanup of per-image temporary prediction files.
        }
    }

    private static void TryKill(Process process)
    {
        try
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
            }
        }
        catch
        {
            // Best-effort cleanup after an unresponsive native prediction worker.
        }
    }

    private static string Shorten(string message)
    {
        var compactMessage = message
            .Replace('\r', ' ')
            .Replace('\n', ' ')
            .Trim();

        return compactMessage.Length <= 500
            ? compactMessage
            : $"{compactMessage[..500]}...";
    }

    private sealed record WorkerBatchResult(
        IReadOnlyDictionary<string, PredictionResponse> Predictions,
        IReadOnlyDictionary<string, string> ItemErrors,
        int? ExitCode,
        string? ProcessError
    );

    private sealed record ParsedBatchResults(
        IReadOnlyDictionary<string, PredictionResponse> Predictions,
        IReadOnlyDictionary<string, string> ItemErrors
    );

    private readonly record struct FailureInfo(int? ExitCode, string Error);
}

public sealed class PredictionProcessException : Exception
{
    public PredictionProcessException(int? exitCode, string message)
        : base(message)
    {
        ExitCode = exitCode;
    }

    public int? ExitCode { get; }
}
