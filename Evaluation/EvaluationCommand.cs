using System.Globalization;
using System.Text;
using System.Text.Json;
using dotnet_api.Services;

namespace dotnet_api.Evaluation;

public static class EvaluationCommand
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
    };

    private static readonly JsonSerializerOptions JsonLineOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false,
    };

    public static bool IsEvaluationCommand(string[] args)
    {
        return args.FirstOrDefault()?.Equals("evaluate", StringComparison.OrdinalIgnoreCase) == true;
    }

    public static bool IsPredictionWorkerCommand(string[] args)
    {
        return args.FirstOrDefault()?.Equals("predict-image", StringComparison.OrdinalIgnoreCase) == true;
    }

    public static bool IsPredictionBatchWorkerCommand(string[] args)
    {
        return args.FirstOrDefault()?.Equals("predict-batch", StringComparison.OrdinalIgnoreCase) == true;
    }

    public static int Run(MlnetPredictionService predictionService, string[] args)
    {
        if (args.Any(arg => arg is "--help" or "-h"))
        {
            PrintUsage();
            return 0;
        }

        if (!TryParseOptions(args, out var options, out var error))
        {
            Console.Error.WriteLine(error);
            Console.Error.WriteLine();
            PrintUsage();
            return 1;
        }

        if (options.IsolatePredictions)
        {
            Console.WriteLine("Using isolated prediction workers.");
        }
        else
        {
            Console.WriteLine("Loading ML.NET model...");
            predictionService.WarmUp(options.ModelId);
        }

        var evaluator = new DetectionEvaluationService(predictionService);
        var run = evaluator.Run(options, Console.WriteLine);

        Console.WriteLine($"Writing evaluation artifacts to {options.OutputDirectory}");
        EvaluationArtifactWriter.Write(run, options.OutputDirectory, options.KeepDataFiles);

        Console.WriteLine(
            string.Format(
                CultureInfo.InvariantCulture,
                "Done. images={0}, skipped={1}, failures={2}, gt={3}, predictions={4}, mAP@0.5={5:0.###}, bestF1={6:0.###} at conf={7:0.###}",
                run.Summary.Images,
                run.Summary.SkippedImages,
                run.Summary.PredictionFailures,
                run.Summary.GroundTruthBoxes,
                run.Summary.Predictions,
                run.Summary.Map50,
                run.Summary.BestF1,
                run.Summary.BestF1Confidence
            )
        );
        return 0;
    }

    public static int RunPredictionWorker(MlnetPredictionService predictionService, string[] args)
    {
        if (!TryParsePredictionWorkerOptions(args, out var imagePath, out var outputPath, out var modelId, out var error))
        {
            Console.Error.WriteLine(error);
            return 1;
        }

        try
        {
            var prediction = predictionService.PredictFile(imagePath, modelId);
            File.WriteAllText(outputPath, JsonSerializer.Serialize(prediction, JsonOptions));
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    public static int RunPredictionBatchWorker(MlnetPredictionService predictionService, string[] args)
    {
        if (!TryParsePredictionBatchWorkerOptions(args, out var manifestPath, out var outputPath, out var error))
        {
            Console.Error.WriteLine(error);
            return 1;
        }

        try
        {
            var manifest = JsonSerializer.Deserialize<PredictionBatchManifest>(
                File.ReadAllText(manifestPath),
                JsonOptions
            );

            if (manifest is null)
            {
                Console.Error.WriteLine("Prediction batch manifest is empty.");
                return 1;
            }

            using var stream = new FileStream(outputPath, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var writer = new StreamWriter(stream, new UTF8Encoding(false))
            {
                AutoFlush = true,
            };

            predictionService.WarmUp(manifest.ModelId);

            foreach (var item in manifest.Images)
            {
                Console.WriteLine($"Evaluating {item.Index}/{item.Total}: {item.FileName}");
                Console.Out.Flush();

                try
                {
                    var prediction = predictionService.PredictFile(item.ImagePath, manifest.ModelId);
                    WritePredictionBatchResult(writer, item, prediction, null);
                }
                catch (Exception exception)
                {
                    var message = Shorten(exception.Message);
                    Console.Error.WriteLine($"Failed {item.FileName}: {message}");
                    WritePredictionBatchResult(writer, item, null, message);
                }
            }

            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static bool TryParseOptions(
        string[] args,
        out EvaluationOptions options,
        out string error
    )
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var booleanFlags = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "in-process",
            "keep-data",
        };

        for (var index = 0; index < args.Length; index++)
        {
            var arg = args[index];
            if (!arg.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = arg[2..];
            if (string.IsNullOrWhiteSpace(key))
            {
                continue;
            }

            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            {
                if (booleanFlags.Contains(key))
                {
                    values[key] = "true";
                    continue;
                }

                error = $"Missing value for --{key}.";
                options = default!;
                return false;
            }

            values[key] = args[++index];
        }

        var imagesDirectory = GetRequiredPath(values, "images");
        var annotationsPath = GetRequiredPath(values, "annotations");

        if (string.IsNullOrWhiteSpace(imagesDirectory))
        {
            error = "Missing required --images path.";
            options = default!;
            return false;
        }

        if (string.IsNullOrWhiteSpace(annotationsPath))
        {
            error = "Missing required --annotations path.";
            options = default!;
            return false;
        }

        imagesDirectory = Path.GetFullPath(imagesDirectory);
        annotationsPath = Path.GetFullPath(annotationsPath);

        if (!Directory.Exists(imagesDirectory))
        {
            error = $"Images directory was not found: {imagesDirectory}";
            options = default!;
            return false;
        }

        if (!File.Exists(annotationsPath))
        {
            error = $"Annotations file was not found: {annotationsPath}";
            options = default!;
            return false;
        }

        var outputDirectory = Path.GetFullPath(
            values.TryGetValue("out", out var output)
                ? output
                : Path.Combine("runs", "half-annotated", "mlnet_eval")
        );
        var modelId = values.TryGetValue("model", out var configuredModelId)
            ? configuredModelId
            : null;

        if (!TryReadDouble(values, "iou", 0.5, out var iouThreshold, out error))
        {
            options = default!;
            return false;
        }

        if (!TryReadDouble(values, "conf", 0.25, out var confidenceThreshold, out error))
        {
            options = default!;
            return false;
        }

        if (!TryReadInt(values, "max-images", out var maxImages, out error))
        {
            options = default!;
            return false;
        }

        if (!TryReadIntValue(values, "batch-size", 10, 1, out var predictionBatchSize, out error))
        {
            options = default!;
            return false;
        }

        if (!TryReadIntValue(values, "timeout-seconds", 120, 1, out var predictionTimeoutSeconds, out error))
        {
            options = default!;
            return false;
        }

        if (!TryReadIntValue(values, "retries", 1, 0, out var predictionRetries, out error))
        {
            options = default!;
            return false;
        }

        options = new EvaluationOptions(
            imagesDirectory,
            annotationsPath,
            outputDirectory,
            string.IsNullOrWhiteSpace(modelId) ? null : modelId,
            Math.Clamp(iouThreshold, 0.01, 0.99),
            Math.Clamp(confidenceThreshold, 0, 1),
            maxImages,
            !values.ContainsKey("in-process"),
            values.ContainsKey("keep-data"),
            predictionBatchSize,
            predictionTimeoutSeconds,
            predictionRetries
        );
        error = string.Empty;
        return true;
    }

    private static string? GetRequiredPath(IReadOnlyDictionary<string, string> values, string key)
    {
        return values.TryGetValue(key, out var value) ? value : null;
    }

    private static bool TryReadDouble(
        IReadOnlyDictionary<string, string> values,
        string key,
        double fallback,
        out double value,
        out string error
    )
    {
        if (!values.TryGetValue(key, out var rawValue))
        {
            value = fallback;
            error = string.Empty;
            return true;
        }

        if (double.TryParse(rawValue, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
        {
            error = string.Empty;
            return true;
        }

        error = $"Invalid --{key} value: {rawValue}";
        return false;
    }

    private static bool TryReadInt(
        IReadOnlyDictionary<string, string> values,
        string key,
        out int? value,
        out string error
    )
    {
        if (!values.TryGetValue(key, out var rawValue))
        {
            value = null;
            error = string.Empty;
            return true;
        }

        if (
            int.TryParse(rawValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedValue)
            && parsedValue > 0
        )
        {
            value = parsedValue;
            error = string.Empty;
            return true;
        }

        value = null;
        error = $"Invalid --{key} value: {rawValue}";
        return false;
    }

    private static bool TryReadIntValue(
        IReadOnlyDictionary<string, string> values,
        string key,
        int fallback,
        int minimum,
        out int value,
        out string error
    )
    {
        if (!values.TryGetValue(key, out var rawValue))
        {
            value = fallback;
            error = string.Empty;
            return true;
        }

        if (
            int.TryParse(rawValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedValue)
            && parsedValue >= minimum
        )
        {
            value = parsedValue;
            error = string.Empty;
            return true;
        }

        value = fallback;
        error = $"Invalid --{key} value: {rawValue}";
        return false;
    }

    private static bool TryParsePredictionWorkerOptions(
        string[] args,
        out string imagePath,
        out string outputPath,
        out string? modelId,
        out string error
    )
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var index = 0; index < args.Length; index++)
        {
            var arg = args[index];
            if (!arg.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = arg[2..];
            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            {
                imagePath = string.Empty;
                outputPath = string.Empty;
                modelId = null;
                error = $"Missing value for --{key}.";
                return false;
            }

            values[key] = args[++index];
        }

        imagePath = values.TryGetValue("image", out var image) ? image : string.Empty;
        outputPath = values.TryGetValue("out", out var output) ? output : string.Empty;
        modelId = values.TryGetValue("model", out var model) ? model : null;

        if (string.IsNullOrWhiteSpace(imagePath))
        {
            error = "Missing required --image path.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            error = "Missing required --out path.";
            return false;
        }

        if (!File.Exists(imagePath))
        {
            error = $"Image file was not found: {imagePath}";
            return false;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".");
        error = string.Empty;
        return true;
    }

    private static bool TryParsePredictionBatchWorkerOptions(
        string[] args,
        out string manifestPath,
        out string outputPath,
        out string error
    )
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var index = 0; index < args.Length; index++)
        {
            var arg = args[index];
            if (!arg.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = arg[2..];
            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            {
                manifestPath = string.Empty;
                outputPath = string.Empty;
                error = $"Missing value for --{key}.";
                return false;
            }

            values[key] = args[++index];
        }

        manifestPath = values.TryGetValue("manifest", out var manifest) ? manifest : string.Empty;
        outputPath = values.TryGetValue("out", out var output) ? output : string.Empty;

        if (string.IsNullOrWhiteSpace(manifestPath))
        {
            error = "Missing required --manifest path.";
            return false;
        }

        if (string.IsNullOrWhiteSpace(outputPath))
        {
            error = "Missing required --out path.";
            return false;
        }

        if (!File.Exists(manifestPath))
        {
            error = $"Prediction batch manifest was not found: {manifestPath}";
            return false;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".");
        error = string.Empty;
        return true;
    }

    private static void WritePredictionBatchResult(
        TextWriter writer,
        PredictionBatchItem item,
        dotnet_api.Responses.PredictionResponse? prediction,
        string? error
    )
    {
        writer.WriteLine(
            JsonSerializer.Serialize(
                new PredictionBatchItemResult(
                    item.ImageId,
                    item.FileName,
                    item.ImagePath,
                    prediction,
                    error
                ),
                JsonLineOptions
            )
        );
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

    private static void PrintUsage()
    {
        Console.WriteLine(
            """
            Usage:
              dotnet run -- evaluate --images <valid-folder> --annotations <annotations.coco> [options]

            Options:
              --out <folder>       Output folder. Default: runs/half-annotated/mlnet_eval
              --model <id>         Optional model id from /models.
              --iou <number>       IoU threshold for TP/FP/FN. Default: 0.5
              --conf <number>      Confidence threshold for matches/confusion matrix. Default: 0.25
              --max-images <n>     Optional limit for a faster smoke test.
              --in-process         Faster old mode; can crash if TorchSharp native runtime asserts.
              --batch-size <n>     Images per isolated worker. Default: 10
              --timeout-seconds <n> Timeout per isolated worker. Default: 120
              --retries <n>        Individual retries after a worker failure. Default: 1
              --keep-data          Also write CSV/JSON debug files.

            Outputs:
              BoxF1_curve.png, BoxP_curve.png, BoxR_curve.png, BoxPR_curve.png,
              confusion_matrix.png, confusion_matrix_normalized.png, labels.jpg
            """
        );
    }
}
