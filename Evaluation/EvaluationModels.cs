using System.Globalization;
using dotnet_api.Responses;

namespace dotnet_api.Evaluation;

public sealed record EvaluationOptions(
    string ImagesDirectory,
    string AnnotationsPath,
    string OutputDirectory,
    string? ModelId,
    double IouThreshold,
    double ConfusionConfidenceThreshold,
    int? MaxImages,
    bool IsolatePredictions,
    bool KeepDataFiles,
    int PredictionBatchSize,
    int PredictionTimeoutSeconds,
    int PredictionRetries
);

public sealed record CocoDataset(
    IReadOnlyList<CocoImage> Images,
    IReadOnlyList<CocoAnnotation> Annotations,
    IReadOnlyList<string> Labels
);

public sealed record CocoImage(
    string Id,
    string FileName,
    int Width,
    int Height,
    string? ResolvedPath
);

public sealed record CocoAnnotation(
    string ImageId,
    string Label,
    DetectionBox Box,
    double Area
);

public sealed record ImageEvaluationResult(
    string ImageId,
    string FileName,
    string ImagePath,
    int Width,
    int Height,
    IReadOnlyList<CocoAnnotation> GroundTruth,
    IReadOnlyList<PredictionAnnotation> Predictions
);

public sealed record PredictionAnnotation(
    string ImageId,
    string FileName,
    string Label,
    double Score,
    DetectionBox Box
);

public sealed record MatchRecord(
    string ImageId,
    string FileName,
    string TrueLabel,
    string PredictedLabel,
    double Score,
    double Iou,
    string Result
);

public sealed record MetricPoint(
    string Label,
    double Confidence,
    int TruePositive,
    int FalsePositive,
    int FalseNegative,
    double Precision,
    double Recall,
    double F1
);

public sealed record PrecisionRecallPoint(
    string Label,
    double Recall,
    double Precision,
    double Confidence
);

public sealed record AveragePrecisionResult(
    string Label,
    double AveragePrecision,
    IReadOnlyList<PrecisionRecallPoint> Curve
);

public sealed record EvaluationSummary(
    int Images,
    int SkippedImages,
    int PredictionFailures,
    int GroundTruthBoxes,
    int Predictions,
    double IouThreshold,
    double ConfusionConfidenceThreshold,
    string? ModelId,
    double BestF1,
    double BestF1Confidence,
    double Map50,
    IReadOnlyDictionary<string, int> GroundTruthByClass,
    IReadOnlyDictionary<string, int> PredictionsByClass
);

public sealed record ConfusionMatrixResult(
    IReadOnlyList<string> Labels,
    int[,] Counts,
    double[,] Normalized
);

public sealed record EvaluationRun(
    CocoDataset Dataset,
    IReadOnlyList<ImageEvaluationResult> Images,
    IReadOnlyList<PredictionFailure> Failures,
    IReadOnlyList<MatchRecord> Matches,
    IReadOnlyList<MetricPoint> Metrics,
    IReadOnlyList<AveragePrecisionResult> AveragePrecisions,
    ConfusionMatrixResult ConfusionMatrix,
    EvaluationSummary Summary
);

public sealed record PredictionFailure(
    string ImageId,
    string FileName,
    string ImagePath,
    int? ExitCode,
    string Error
);

public sealed record PredictionBatchManifest(
    string? ModelId,
    IReadOnlyList<PredictionBatchItem> Images
);

public sealed record PredictionBatchItem(
    int Index,
    int Total,
    string ImageId,
    string FileName,
    string ImagePath
);

public sealed record PredictionBatchItemResult(
    string ImageId,
    string FileName,
    string ImagePath,
    PredictionResponse? Prediction,
    string? Error
);

public sealed record PredictionBatchRunResult(
    IReadOnlyDictionary<string, PredictionResponse> Predictions,
    IReadOnlyList<PredictionFailure> Failures
);

public readonly record struct DetectionBox(double X1, double Y1, double X2, double Y2)
{
    public double Width => Math.Max(0, X2 - X1);

    public double Height => Math.Max(0, Y2 - Y1);

    public double Area => Width * Height;

    public double CenterX => X1 + Width / 2;

    public double CenterY => Y1 + Height / 2;

    public static DetectionBox FromCoco(double x, double y, double width, double height)
    {
        return new DetectionBox(x, y, x + width, y + height);
    }

    public DetectionBox Clamp(int imageWidth, int imageHeight)
    {
        var x1 = Math.Clamp(X1, 0, imageWidth);
        var y1 = Math.Clamp(Y1, 0, imageHeight);
        var x2 = Math.Clamp(X2, 0, imageWidth);
        var y2 = Math.Clamp(Y2, 0, imageHeight);

        if (x2 < x1)
        {
            (x1, x2) = (x2, x1);
        }

        if (y2 < y1)
        {
            (y1, y2) = (y2, y1);
        }

        return new DetectionBox(x1, y1, x2, y2);
    }

    public double Iou(DetectionBox other)
    {
        var intersectionX1 = Math.Max(X1, other.X1);
        var intersectionY1 = Math.Max(Y1, other.Y1);
        var intersectionX2 = Math.Min(X2, other.X2);
        var intersectionY2 = Math.Min(Y2, other.Y2);
        var intersectionWidth = Math.Max(0, intersectionX2 - intersectionX1);
        var intersectionHeight = Math.Max(0, intersectionY2 - intersectionY1);
        var intersectionArea = intersectionWidth * intersectionHeight;
        var unionArea = Area + other.Area - intersectionArea;

        return unionArea <= 0 ? 0 : intersectionArea / unionArea;
    }

    public string ToCsv()
    {
        return string.Join(
            ",",
            X1.ToString("0.###", CultureInfo.InvariantCulture),
            Y1.ToString("0.###", CultureInfo.InvariantCulture),
            X2.ToString("0.###", CultureInfo.InvariantCulture),
            Y2.ToString("0.###", CultureInfo.InvariantCulture)
        );
    }
}
