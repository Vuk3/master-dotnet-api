using dotnet_api.Responses;
using dotnet_api.Services;

namespace dotnet_api.Evaluation;

public sealed class DetectionEvaluationService
{
    private const string AllClassesLabel = "all";
    private const string BackgroundLabel = "background";
    private readonly MlnetPredictionService predictionService;

    public DetectionEvaluationService(MlnetPredictionService predictionService)
    {
        this.predictionService = predictionService;
    }

    public EvaluationRun Run(EvaluationOptions options, Action<string>? progress = null)
    {
        var dataset = CocoDatasetLoader.Load(options.AnnotationsPath, options.ImagesDirectory);
        var annotationsByImage = dataset.Annotations
            .GroupBy(annotation => annotation.ImageId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.ToList(), StringComparer.OrdinalIgnoreCase);
        var selectedImages = options.MaxImages is { } maxImages
            ? dataset.Images.Take(maxImages).ToList()
            : dataset.Images.ToList();
        var imageResults = new List<ImageEvaluationResult>();
        var failures = new List<PredictionFailure>();
        var skippedImages = 0;
        var isolatedPredictions = new Dictionary<string, PredictionResponse>(StringComparer.OrdinalIgnoreCase);
        var isolatedFailures = new Dictionary<string, PredictionFailure>(StringComparer.OrdinalIgnoreCase);

        if (options.IsolatePredictions)
        {
            var predictionItems = new List<PredictionBatchItem>();

            for (var index = 0; index < selectedImages.Count; index++)
            {
                var image = selectedImages[index];
                if (string.IsNullOrWhiteSpace(image.ResolvedPath) || !File.Exists(image.ResolvedPath))
                {
                    continue;
                }

                predictionItems.Add(
                    new PredictionBatchItem(
                        index + 1,
                        selectedImages.Count,
                        image.Id,
                        image.FileName,
                        image.ResolvedPath
                    )
                );
            }

            var batchRun = ImagePredictionProcessRunner.PredictBatch(predictionItems, options, progress);
            isolatedPredictions = batchRun.Predictions.ToDictionary(
                pair => pair.Key,
                pair => pair.Value,
                StringComparer.OrdinalIgnoreCase
            );

            foreach (var failure in batchRun.Failures)
            {
                isolatedFailures[failure.ImageId] = failure;
            }
        }

        for (var index = 0; index < selectedImages.Count; index++)
        {
            var image = selectedImages[index];
            if (string.IsNullOrWhiteSpace(image.ResolvedPath) || !File.Exists(image.ResolvedPath))
            {
                skippedImages++;
                continue;
            }

            PredictionResponse? prediction;

            if (options.IsolatePredictions)
            {
                if (!isolatedPredictions.TryGetValue(image.Id, out prediction))
                {
                    var failure = isolatedFailures.TryGetValue(image.Id, out var isolatedFailure)
                        ? isolatedFailure
                        : new PredictionFailure(
                            image.Id,
                            image.FileName,
                            image.ResolvedPath,
                            null,
                            "Prediction worker did not return a result for this image."
                        );

                    failures.Add(failure);
                    skippedImages++;
                    progress?.Invoke($"Skipped {image.FileName}: {failure.Error}");
                    continue;
                }
            }
            else
            {
                progress?.Invoke($"Evaluating {index + 1}/{selectedImages.Count}: {image.FileName}");
                prediction = TryPredictInProcess(
                    image.ResolvedPath,
                    image.FileName,
                    options,
                    progress,
                    out var failure
                );

                if (prediction is null)
                {
                    failures.Add(
                        new PredictionFailure(
                            image.Id,
                            image.FileName,
                            image.ResolvedPath,
                            failure.ExitCode,
                            failure.Error
                        )
                    );
                    skippedImages++;
                    progress?.Invoke($"Skipped {image.FileName}: {failure.Error}");
                    continue;
                }
            }

            var width = prediction.ImageWidth > 0 ? prediction.ImageWidth : image.Width;
            var height = prediction.ImageHeight > 0 ? prediction.ImageHeight : image.Height;
            annotationsByImage.TryGetValue(image.Id, out var imageAnnotations);

            var groundTruth = (imageAnnotations ?? [])
                .Select(annotation => annotation with { Box = annotation.Box.Clamp(width, height) })
                .Where(annotation => annotation.Box.Area > 0)
                .ToList();
            var predictions = prediction.Detections
                .Select(detection => new PredictionAnnotation(
                    image.Id,
                    image.FileName,
                    detection.Label.Trim(),
                    detection.Score,
                    new DetectionBox(
                        detection.Box.X1,
                        detection.Box.Y1,
                        detection.Box.X2,
                        detection.Box.Y2
                    ).Clamp(width, height)
                ))
                .Where(prediction => prediction.Box.Area > 0)
                .ToList();

            imageResults.Add(
                new ImageEvaluationResult(
                    image.Id,
                    image.FileName,
                    image.ResolvedPath,
                    width,
                    height,
                    groundTruth,
                    predictions
                )
            );
        }

        var labels = GetEvaluationLabels(dataset, imageResults);
        var metrics = CalculateMetrics(imageResults, labels, options.IouThreshold);
        var averagePrecisions = labels
            .Select(label => CalculateAveragePrecision(imageResults, label, options.IouThreshold))
            .ToList();
        var matches = BuildMatches(
            imageResults,
            labels,
            options.IouThreshold,
            options.ConfusionConfidenceThreshold
        );
        var confusionMatrix = BuildConfusionMatrix(
            imageResults,
            labels,
            options.IouThreshold,
            options.ConfusionConfidenceThreshold
        );
        var allMetrics = metrics
            .Where(metric => metric.Label == AllClassesLabel)
            .OrderByDescending(metric => metric.F1)
            .ThenBy(metric => metric.Confidence)
            .ToList();
        var bestF1 = allMetrics.FirstOrDefault();
        var summary = new EvaluationSummary(
            imageResults.Count,
            skippedImages,
            failures.Count,
            imageResults.Sum(result => result.GroundTruth.Count),
            imageResults.Sum(result => result.Predictions.Count),
            options.IouThreshold,
            options.ConfusionConfidenceThreshold,
            options.ModelId,
            bestF1?.F1 ?? 0,
            bestF1?.Confidence ?? 0,
            averagePrecisions.Count == 0 ? 0 : averagePrecisions.Average(result => result.AveragePrecision),
            CountByLabel(imageResults.SelectMany(result => result.GroundTruth).Select(annotation => annotation.Label)),
            CountByLabel(imageResults.SelectMany(result => result.Predictions).Select(annotation => annotation.Label))
        );

        return new EvaluationRun(
            dataset,
            imageResults,
            failures,
            matches,
            metrics,
            averagePrecisions,
            confusionMatrix,
            summary
        );
    }

    private PredictionResponse? TryPredictInProcess(
        string imagePath,
        string fileName,
        EvaluationOptions options,
        Action<string>? progress,
        out PredictionAttemptFailure failure
    )
    {
        var attempts = Math.Max(1, options.PredictionRetries + 1);
        failure = default;

        for (var attempt = 1; attempt <= attempts; attempt++)
        {
            try
            {
                failure = default;
                return predictionService.PredictFile(imagePath, options.ModelId);
            }
            catch (PredictionProcessException exception)
            {
                failure = new PredictionAttemptFailure(exception.ExitCode, Shorten(exception.Message));
            }
            catch (Exception exception)
            {
                failure = new PredictionAttemptFailure(null, Shorten(exception.Message));
            }

            if (attempt < attempts)
            {
                progress?.Invoke(
                    $"Retrying {fileName} after failure (attempt {attempt + 1}/{attempts})"
                );
            }
        }

        return null;
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

    private static IReadOnlyList<string> GetEvaluationLabels(
        CocoDataset dataset,
        IReadOnlyList<ImageEvaluationResult> imageResults
    )
    {
        var labels = dataset.Labels.Count > 0
            ? dataset.Labels
            : imageResults.SelectMany(result => result.Predictions).Select(prediction => prediction.Label).ToList();

        return labels
            .Where(label => !string.IsNullOrWhiteSpace(label))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(label => label, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private static IReadOnlyList<MetricPoint> CalculateMetrics(
        IReadOnlyList<ImageEvaluationResult> imageResults,
        IReadOnlyList<string> labels,
        double iouThreshold
    )
    {
        var points = new List<MetricPoint>();

        for (var index = 0; index <= 100; index++)
        {
            var confidence = index / 100.0;
            var aggregateTruePositive = 0;
            var aggregateFalsePositive = 0;
            var aggregateFalseNegative = 0;

            foreach (var label in labels)
            {
                var counts = CountClassMatches(imageResults, label, confidence, iouThreshold);
                aggregateTruePositive += counts.TruePositive;
                aggregateFalsePositive += counts.FalsePositive;
                aggregateFalseNegative += counts.FalseNegative;
                points.Add(CreateMetricPoint(label, confidence, counts));
            }

            points.Add(
                CreateMetricPoint(
                    AllClassesLabel,
                    confidence,
                    new MatchCounts(
                        aggregateTruePositive,
                        aggregateFalsePositive,
                        aggregateFalseNegative
                    )
                )
            );
        }

        return points;
    }

    private static MatchCounts CountClassMatches(
        IReadOnlyList<ImageEvaluationResult> imageResults,
        string label,
        double confidence,
        double iouThreshold
    )
    {
        var truePositive = 0;
        var falsePositive = 0;
        var falseNegative = 0;

        foreach (var imageResult in imageResults)
        {
            var groundTruth = imageResult.GroundTruth
                .Where(annotation => IsSameLabel(annotation.Label, label))
                .ToList();
            var predictions = imageResult.Predictions
                .Where(prediction => IsSameLabel(prediction.Label, label) && prediction.Score >= confidence)
                .OrderByDescending(prediction => prediction.Score)
                .ToList();
            var matchedGroundTruth = new bool[groundTruth.Count];

            foreach (var prediction in predictions)
            {
                var bestIndex = FindBestMatch(prediction.Box, groundTruth, matchedGroundTruth);

                if (bestIndex >= 0 && prediction.Box.Iou(groundTruth[bestIndex].Box) >= iouThreshold)
                {
                    matchedGroundTruth[bestIndex] = true;
                    truePositive++;
                }
                else
                {
                    falsePositive++;
                }
            }

            falseNegative += matchedGroundTruth.Count(matched => !matched);
        }

        return new MatchCounts(truePositive, falsePositive, falseNegative);
    }

    private static MetricPoint CreateMetricPoint(
        string label,
        double confidence,
        MatchCounts counts
    )
    {
        var precision = counts.TruePositive + counts.FalsePositive == 0
            ? 0
            : (double)counts.TruePositive / (counts.TruePositive + counts.FalsePositive);
        var recall = counts.TruePositive + counts.FalseNegative == 0
            ? 0
            : (double)counts.TruePositive / (counts.TruePositive + counts.FalseNegative);
        var f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

        return new MetricPoint(
            label,
            confidence,
            counts.TruePositive,
            counts.FalsePositive,
            counts.FalseNegative,
            precision,
            recall,
            f1
        );
    }

    private static AveragePrecisionResult CalculateAveragePrecision(
        IReadOnlyList<ImageEvaluationResult> imageResults,
        string label,
        double iouThreshold
    )
    {
        var totalGroundTruth = imageResults.Sum(
            image => image.GroundTruth.Count(annotation => IsSameLabel(annotation.Label, label))
        );
        var predictions = imageResults
            .SelectMany(image => image.Predictions)
            .Where(prediction => IsSameLabel(prediction.Label, label))
            .OrderByDescending(prediction => prediction.Score)
            .ToList();
        var matchedByImage = imageResults.ToDictionary(
            image => image.ImageId,
            image => new bool[
                image.GroundTruth.Count(annotation => IsSameLabel(annotation.Label, label))
            ],
            StringComparer.OrdinalIgnoreCase
        );
        var groundTruthByImage = imageResults.ToDictionary(
            image => image.ImageId,
            image => image.GroundTruth
                .Where(annotation => IsSameLabel(annotation.Label, label))
                .ToList(),
            StringComparer.OrdinalIgnoreCase
        );
        var truePositive = 0;
        var falsePositive = 0;
        var curve = new List<PrecisionRecallPoint>();
        var recalls = new List<double>();
        var precisions = new List<double>();

        foreach (var prediction in predictions)
        {
            var groundTruth = groundTruthByImage[prediction.ImageId];
            var matched = matchedByImage[prediction.ImageId];
            var bestIndex = FindBestMatch(prediction.Box, groundTruth, matched);

            if (bestIndex >= 0 && prediction.Box.Iou(groundTruth[bestIndex].Box) >= iouThreshold)
            {
                matched[bestIndex] = true;
                truePositive++;
            }
            else
            {
                falsePositive++;
            }

            var recall = totalGroundTruth == 0 ? 0 : (double)truePositive / totalGroundTruth;
            var precision = truePositive + falsePositive == 0
                ? 0
                : (double)truePositive / (truePositive + falsePositive);

            recalls.Add(recall);
            precisions.Add(precision);
            curve.Add(new PrecisionRecallPoint(label, recall, precision, prediction.Score));
        }

        return new AveragePrecisionResult(
            label,
            CalculateAp(recalls, precisions),
            curve
        );
    }

    private static IReadOnlyList<MatchRecord> BuildMatches(
        IReadOnlyList<ImageEvaluationResult> imageResults,
        IReadOnlyList<string> labels,
        double iouThreshold,
        double confidence
    )
    {
        var records = new List<MatchRecord>();

        foreach (var imageResult in imageResults)
        {
            foreach (var label in labels)
            {
                var groundTruth = imageResult.GroundTruth
                    .Where(annotation => IsSameLabel(annotation.Label, label))
                    .ToList();
                var predictions = imageResult.Predictions
                    .Where(prediction => IsSameLabel(prediction.Label, label) && prediction.Score >= confidence)
                    .OrderByDescending(prediction => prediction.Score)
                    .ToList();
                var matchedGroundTruth = new bool[groundTruth.Count];

                foreach (var prediction in predictions)
                {
                    var bestIndex = FindBestMatch(prediction.Box, groundTruth, matchedGroundTruth);

                    if (bestIndex >= 0 && prediction.Box.Iou(groundTruth[bestIndex].Box) >= iouThreshold)
                    {
                        matchedGroundTruth[bestIndex] = true;
                        records.Add(
                            new MatchRecord(
                                imageResult.ImageId,
                                imageResult.FileName,
                                groundTruth[bestIndex].Label,
                                prediction.Label,
                                prediction.Score,
                                prediction.Box.Iou(groundTruth[bestIndex].Box),
                                "TP"
                            )
                        );
                    }
                    else
                    {
                        records.Add(
                            new MatchRecord(
                                imageResult.ImageId,
                                imageResult.FileName,
                                BackgroundLabel,
                                prediction.Label,
                                prediction.Score,
                                bestIndex >= 0 ? prediction.Box.Iou(groundTruth[bestIndex].Box) : 0,
                                "FP"
                            )
                        );
                    }
                }

                for (var index = 0; index < groundTruth.Count; index++)
                {
                    if (!matchedGroundTruth[index])
                    {
                        records.Add(
                            new MatchRecord(
                                imageResult.ImageId,
                                imageResult.FileName,
                                groundTruth[index].Label,
                                BackgroundLabel,
                                0,
                                0,
                                "FN"
                            )
                        );
                    }
                }
            }
        }

        return records;
    }

    private static ConfusionMatrixResult BuildConfusionMatrix(
        IReadOnlyList<ImageEvaluationResult> imageResults,
        IReadOnlyList<string> labels,
        double iouThreshold,
        double confidence
    )
    {
        var matrixLabels = labels.Concat([BackgroundLabel]).ToList();
        var labelIndex = matrixLabels
            .Select((label, index) => new { label, index })
            .ToDictionary(item => item.label, item => item.index, StringComparer.OrdinalIgnoreCase);
        var backgroundIndex = labelIndex[BackgroundLabel];
        var counts = new int[matrixLabels.Count, matrixLabels.Count];

        foreach (var imageResult in imageResults)
        {
            var groundTruth = imageResult.GroundTruth.ToList();
            var matchedGroundTruth = new bool[groundTruth.Count];
            var predictions = imageResult.Predictions
                .Where(prediction => prediction.Score >= confidence)
                .OrderByDescending(prediction => prediction.Score)
                .ToList();

            foreach (var prediction in predictions)
            {
                var bestIndex = FindBestMatch(prediction.Box, groundTruth, matchedGroundTruth);
                var predictedIndex = labelIndex.TryGetValue(prediction.Label, out var foundPredictedIndex)
                    ? foundPredictedIndex
                    : backgroundIndex;

                if (bestIndex >= 0 && prediction.Box.Iou(groundTruth[bestIndex].Box) >= iouThreshold)
                {
                    var trueIndex = labelIndex[groundTruth[bestIndex].Label];
                    counts[predictedIndex, trueIndex]++;
                    matchedGroundTruth[bestIndex] = true;
                }
                else
                {
                    counts[predictedIndex, backgroundIndex]++;
                }
            }

            for (var index = 0; index < groundTruth.Count; index++)
            {
                if (!matchedGroundTruth[index])
                {
                    counts[backgroundIndex, labelIndex[groundTruth[index].Label]]++;
                }
            }
        }

        var normalized = new double[matrixLabels.Count, matrixLabels.Count];
        for (var column = 0; column < matrixLabels.Count; column++)
        {
            var columnTotal = 0;
            for (var row = 0; row < matrixLabels.Count; row++)
            {
                columnTotal += counts[row, column];
            }

            if (columnTotal == 0)
            {
                continue;
            }

            for (var row = 0; row < matrixLabels.Count; row++)
            {
                normalized[row, column] = (double)counts[row, column] / columnTotal;
            }
        }

        return new ConfusionMatrixResult(matrixLabels, counts, normalized);
    }

    private static int FindBestMatch<TAnnotation>(
        DetectionBox predictionBox,
        IReadOnlyList<TAnnotation> groundTruth,
        IReadOnlyList<bool> matchedGroundTruth
    )
        where TAnnotation : notnull
    {
        var bestIndex = -1;
        var bestIou = 0.0;

        for (var index = 0; index < groundTruth.Count; index++)
        {
            if (matchedGroundTruth[index])
            {
                continue;
            }

            var groundTruthBox = groundTruth[index] switch
            {
                CocoAnnotation annotation => annotation.Box,
                _ => throw new InvalidOperationException("Unsupported annotation type."),
            };
            var iou = predictionBox.Iou(groundTruthBox);

            if (iou > bestIou)
            {
                bestIou = iou;
                bestIndex = index;
            }
        }

        return bestIndex;
    }

    private static double CalculateAp(IReadOnlyList<double> recalls, IReadOnlyList<double> precisions)
    {
        if (recalls.Count == 0 || precisions.Count == 0)
        {
            return 0;
        }

        var mrec = new List<double> { 0 };
        mrec.AddRange(recalls);
        mrec.Add(1);

        var mpre = new List<double> { 1 };
        mpre.AddRange(precisions);
        mpre.Add(0);

        for (var index = mpre.Count - 2; index >= 0; index--)
        {
            mpre[index] = Math.Max(mpre[index], mpre[index + 1]);
        }

        var ap = 0.0;
        for (var index = 0; index < mrec.Count - 1; index++)
        {
            if (Math.Abs(mrec[index + 1] - mrec[index]) > double.Epsilon)
            {
                ap += (mrec[index + 1] - mrec[index]) * mpre[index + 1];
            }
        }

        return ap;
    }

    private static IReadOnlyDictionary<string, int> CountByLabel(IEnumerable<string> labels)
    {
        return labels
            .Where(label => !string.IsNullOrWhiteSpace(label))
            .GroupBy(label => label, StringComparer.OrdinalIgnoreCase)
            .OrderBy(group => group.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);
    }

    private static bool IsSameLabel(string left, string right)
    {
        return string.Equals(left.Trim(), right.Trim(), StringComparison.OrdinalIgnoreCase);
    }

    private readonly record struct MatchCounts(
        int TruePositive,
        int FalsePositive,
        int FalseNegative
    );

    private readonly record struct PredictionAttemptFailure(
        int? ExitCode,
        string Error
    );
}
