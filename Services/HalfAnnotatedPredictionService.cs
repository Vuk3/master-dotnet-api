using Dotnet_api;
using dotnet_api.Responses;
using Microsoft.ML.Data;
using System.Diagnostics;

namespace dotnet_api.Services;

public class HalfAnnotatedPredictionService
{
    private readonly ILogger<HalfAnnotatedPredictionService> logger;
    private readonly DotnetModelCatalogService modelCatalogService;
    private readonly object predictionLock = new();

    public HalfAnnotatedPredictionService(
        ILogger<HalfAnnotatedPredictionService> logger,
        DotnetModelCatalogService modelCatalogService
    )
    {
        this.logger = logger;
        this.modelCatalogService = modelCatalogService;
    }

    public void WarmUp()
    {
        try
        {
            var stopwatch = Stopwatch.StartNew();
            lock (predictionLock)
            {
                _ = HalfAnnotatedModel.PredictEngine.Value;
            }

            logger.LogInformation(
                "ML.NET prediction engine loaded in {ElapsedMilliseconds} ms",
                stopwatch.ElapsedMilliseconds
            );
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "ML.NET prediction engine warm-up failed");
        }
    }

    public async Task<PredictionResponse> Predict(IFormFile file, string? modelId)
    {
        await using var imageStream = new MemoryStream();
        await file.CopyToAsync(imageStream);
        imageStream.Position = 0;

        var image = MLImage.CreateFromStream(imageStream);
        var input = new HalfAnnotatedModel.ModelInput
        {
            Image = image,
        };

        HalfAnnotatedModel.ModelOutput output;
        lock (predictionLock)
        {
            output = HalfAnnotatedModel.Predict(input);
        }

        var selectedModel = modelCatalogService.GetModel(modelId);

        return new PredictionResponse(
            selectedModel?.Name ?? "ML.NET Object Detection",
            selectedModel?.Id,
            selectedModel?.AnnotationType,
            image.Width,
            image.Height,
            MapDetections(output, image.Width, image.Height),
            file.FileName,
            file.ContentType
        );
    }

    private static IReadOnlyList<DetectionResponse> MapDetections(
        HalfAnnotatedModel.ModelOutput output,
        int imageWidth,
        int imageHeight
    )
    {
        var labels = output.PredictedLabel ?? Array.Empty<string>();
        var scores = output.Score ?? Array.Empty<float>();
        var boxes = output.PredictedBoundingBoxes ?? Array.Empty<float>();
        var detectionCount = Math.Min(labels.Length, Math.Min(scores.Length, boxes.Length / 4));
        var detections = new List<DetectionResponse>(detectionCount);

        for (var index = 0; index < detectionCount; index++)
        {
            var boxIndex = index * 4;
            var box = MapBoundingBox(boxes, boxIndex, imageWidth, imageHeight);
            if (box == null)
            {
                continue;
            }

            detections.Add(
                new DetectionResponse(
                    labels[index],
                    NormalizeScore(scores[index]),
                    box
                )
            );
        }

        return detections;
    }

    private static BoundingBoxResponse? MapBoundingBox(
        float[] boxes,
        int boxIndex,
        int imageWidth,
        int imageHeight
    )
    {
        var x1 = boxes[boxIndex];
        var y1 = boxes[boxIndex + 1];
        var x2 = boxes[boxIndex + 2];
        var y2 = boxes[boxIndex + 3];

        if (!IsFinite(x1) || !IsFinite(y1) || !IsFinite(x2) || !IsFinite(y2))
        {
            return null;
        }

        var isNormalized = Math.Abs(x1) <= 1.5f
            && Math.Abs(y1) <= 1.5f
            && Math.Abs(x2) <= 1.5f
            && Math.Abs(y2) <= 1.5f;

        if (isNormalized)
        {
            x1 *= imageWidth;
            x2 *= imageWidth;
            y1 *= imageHeight;
            y2 *= imageHeight;
        }

        if (x2 <= x1 && x2 > 0)
        {
            x2 += x1;
        }

        if (y2 <= y1 && y2 > 0)
        {
            y2 += y1;
        }

        x1 = Clamp(x1, 0, imageWidth);
        y1 = Clamp(y1, 0, imageHeight);
        x2 = Clamp(x2, 0, imageWidth);
        y2 = Clamp(y2, 0, imageHeight);

        if (x2 < x1)
        {
            (x1, x2) = (x2, x1);
        }

        if (y2 < y1)
        {
            (y1, y2) = (y2, y1);
        }

        if (x2 - x1 < 1 || y2 - y1 < 1)
        {
            return null;
        }

        return new BoundingBoxResponse(x1, y1, x2, y2);
    }

    private static float NormalizeScore(float score)
    {
        return score > 1 ? score / 100 : score;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static float Clamp(float value, float min, float max)
    {
        return Math.Min(max, Math.Max(min, value));
    }
}
