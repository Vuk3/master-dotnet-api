using System.Globalization;
using System.Text;
using System.Text.Json;

namespace dotnet_api.Evaluation;

public static class EvaluationArtifactWriter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
    };

    public static void Write(EvaluationRun run, string outputDirectory, bool keepDataFiles)
    {
        Directory.CreateDirectory(outputDirectory);

        if (keepDataFiles)
        {
            WritePredictionsJsonl(run, Path.Combine(outputDirectory, "predictions.jsonl"));
            WritePredictionFailuresCsv(run, Path.Combine(outputDirectory, "prediction_failures.csv"));
            WriteGroundTruthCsv(run, Path.Combine(outputDirectory, "ground_truth.csv"));
            WritePredictionsCsv(run, Path.Combine(outputDirectory, "predictions.csv"));
            WriteMatchesCsv(run, Path.Combine(outputDirectory, "matches.csv"));
            WriteMetricsCsv(run, Path.Combine(outputDirectory, "metrics_by_threshold.csv"));
            WriteAveragePrecisionCsv(run, Path.Combine(outputDirectory, "average_precision.csv"));
            WriteConfusionMatrixCsv(run.ConfusionMatrix, Path.Combine(outputDirectory, "confusion_matrix.csv"), false);
            WriteConfusionMatrixCsv(
                run.ConfusionMatrix,
                Path.Combine(outputDirectory, "confusion_matrix_normalized.csv"),
                true
            );
            File.WriteAllText(
                Path.Combine(outputDirectory, "summary.json"),
                JsonSerializer.Serialize(run.Summary, JsonOptions),
                Encoding.UTF8
            );
        }

        EvaluationPlotWriter.Write(run, outputDirectory);
    }

    private static void WritePredictionsJsonl(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        var jsonOptions = new JsonSerializerOptions(JsonOptions)
        {
            WriteIndented = false,
        };

        foreach (var image in run.Images)
        {
            writer.WriteLine(JsonSerializer.Serialize(image, jsonOptions));
        }
    }

    private static void WritePredictionFailuresCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("image_id,file_name,image_path,exit_code,error");

        foreach (var failure in run.Failures)
        {
            writer.WriteLine(
                Csv(
                    failure.ImageId,
                    failure.FileName,
                    failure.ImagePath,
                    failure.ExitCode,
                    failure.Error
                )
            );
        }
    }

    private static void WriteGroundTruthCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("image_id,file_name,label,x1,y1,x2,y2,area");

        foreach (var image in run.Images)
        {
            foreach (var annotation in image.GroundTruth)
            {
                writer.WriteLine(
                    Csv(
                        image.ImageId,
                        image.FileName,
                        annotation.Label,
                        annotation.Box.X1,
                        annotation.Box.Y1,
                        annotation.Box.X2,
                        annotation.Box.Y2,
                        annotation.Box.Area
                    )
                );
            }
        }
    }

    private static void WritePredictionsCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("image_id,file_name,label,score,x1,y1,x2,y2,area");

        foreach (var image in run.Images)
        {
            foreach (var prediction in image.Predictions)
            {
                writer.WriteLine(
                    Csv(
                        image.ImageId,
                        image.FileName,
                        prediction.Label,
                        prediction.Score,
                        prediction.Box.X1,
                        prediction.Box.Y1,
                        prediction.Box.X2,
                        prediction.Box.Y2,
                        prediction.Box.Area
                    )
                );
            }
        }
    }

    private static void WriteMatchesCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("image_id,file_name,true_label,predicted_label,score,iou,result");

        foreach (var match in run.Matches)
        {
            writer.WriteLine(
                Csv(
                    match.ImageId,
                    match.FileName,
                    match.TrueLabel,
                    match.PredictedLabel,
                    match.Score,
                    match.Iou,
                    match.Result
                )
            );
        }
    }

    private static void WriteMetricsCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("label,confidence,tp,fp,fn,precision,recall,f1");

        foreach (var metric in run.Metrics)
        {
            writer.WriteLine(
                Csv(
                    metric.Label,
                    metric.Confidence,
                    metric.TruePositive,
                    metric.FalsePositive,
                    metric.FalseNegative,
                    metric.Precision,
                    metric.Recall,
                    metric.F1
                )
            );
        }
    }

    private static void WriteAveragePrecisionCsv(EvaluationRun run, string outputPath)
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine("label,ap50");

        foreach (var result in run.AveragePrecisions)
        {
            writer.WriteLine(Csv(result.Label, result.AveragePrecision));
        }
    }

    private static void WriteConfusionMatrixCsv(
        ConfusionMatrixResult matrix,
        string outputPath,
        bool normalized
    )
    {
        using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
        writer.WriteLine(Csv(["predicted/true", .. matrix.Labels]));

        for (var row = 0; row < matrix.Labels.Count; row++)
        {
            var values = new List<object?> { matrix.Labels[row] };
            for (var column = 0; column < matrix.Labels.Count; column++)
            {
                values.Add(normalized ? matrix.Normalized[row, column] : matrix.Counts[row, column]);
            }

            writer.WriteLine(Csv(values));
        }
    }

    private static string Csv(params object?[] values)
    {
        return Csv((IEnumerable<object?>)values);
    }

    private static string Csv(IEnumerable<object?> values)
    {
        return string.Join(",", values.Select(FormatCsvValue));
    }

    private static string FormatCsvValue(object? value)
    {
        var text = value switch
        {
            null => string.Empty,
            double number => number.ToString("0.######", CultureInfo.InvariantCulture),
            float number => number.ToString("0.######", CultureInfo.InvariantCulture),
            decimal number => number.ToString("0.######", CultureInfo.InvariantCulture),
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? string.Empty,
        };

        return text.Contains(',') || text.Contains('"') || text.Contains('\n') || text.Contains('\r')
            ? $"\"{text.Replace("\"", "\"\"")}\""
            : text;
    }
}
