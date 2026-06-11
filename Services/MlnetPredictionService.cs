using Dotnet_api;
using dotnet_api.Responses;
using Microsoft.ML.Data;
using System.Diagnostics;

namespace dotnet_api.Services;

public class MlnetPredictionService
{
    private readonly ILogger<MlnetPredictionService> logger;
    private readonly DotnetModelCatalogService modelCatalogService;
    private readonly object predictionLock = new();

    public MlnetPredictionService(
        ILogger<MlnetPredictionService> logger,
        DotnetModelCatalogService modelCatalogService
    )
    {
        this.logger = logger;
        this.modelCatalogService = modelCatalogService;
    }

    public void WarmUp()
    {
        WarmUp(null);
    }

    public void WarmUp(string? modelId)
    {
        try
        {
            var selectedModel = GetRequiredModel(modelId);
            var stopwatch = Stopwatch.StartNew();
            lock (predictionLock)
            {
                switch (GetMlnetModelKind(selectedModel))
                {
                    case MlnetModelKind.FullyAnnotated:
                        _ = FullyAnnotatedModel.PredictEngine.Value;
                        break;
                    case MlnetModelKind.HalfAnnotated:
                    default:
                        _ = HalfAnnotatedModel.PredictEngine.Value;
                        break;
                }
            }

            logger.LogInformation(
                "ML.NET prediction engine {ModelId} loaded in {ElapsedMilliseconds} ms",
                selectedModel.Id,
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

        return PredictStream(
            imageStream,
            modelId,
            file.FileName,
            file.ContentType
        );
    }

    public PredictionResponse PredictFile(string imagePath, string? modelId)
    {
        using var imageStream = File.OpenRead(imagePath);

        return PredictStream(
            imageStream,
            modelId,
            Path.GetFileName(imagePath),
            GetContentType(imagePath)
        );
    }

    private PredictionResponse PredictStream(
        Stream imageStream,
        string? modelId,
        string? fileName,
        string? contentType
    )
    {
        var selectedModel = GetRequiredModel(modelId);
        var image = MLImage.CreateFromStream(imageStream);
        var output = PredictSelectedModel(image, selectedModel);

        return new PredictionResponse(
            selectedModel.Name,
            selectedModel.Id,
            selectedModel.AnnotationType,
            image.Width,
            image.Height,
            MapDetections(output, image.Width, image.Height),
            fileName,
            contentType
        );
    }

    private ModelOptionResponse GetRequiredModel(string? modelId)
    {
        var selectedModel = modelCatalogService.GetModel(modelId);
        if (selectedModel is not null)
        {
            return selectedModel;
        }

        throw new ArgumentException($"Requested .NET model was not found: {modelId}");
    }

    private ModelOutputAdapter PredictSelectedModel(MLImage image, ModelOptionResponse selectedModel)
    {
        lock (predictionLock)
        {
            return GetMlnetModelKind(selectedModel) switch
            {
                MlnetModelKind.FullyAnnotated => PredictFullyAnnotated(image),
                MlnetModelKind.HalfAnnotated => PredictHalfAnnotated(image),
                _ => PredictHalfAnnotated(image),
            };
        }
    }

    private static ModelOutputAdapter PredictHalfAnnotated(MLImage image)
    {
        var input = new HalfAnnotatedModel.ModelInput
        {
            Image = image,
        };
        var output = HalfAnnotatedModel.Predict(input);

        return new ModelOutputAdapter(
            output.PredictedLabel ?? Array.Empty<string>(),
            output.Score ?? Array.Empty<float>(),
            output.PredictedBoundingBoxes ?? Array.Empty<float>()
        );
    }

    private static ModelOutputAdapter PredictFullyAnnotated(MLImage image)
    {
        var input = new FullyAnnotatedModel.ModelInput
        {
            Image = image,
        };
        var output = FullyAnnotatedModel.Predict(input);

        return new ModelOutputAdapter(
            output.PredictedLabel ?? Array.Empty<string>(),
            output.Score ?? Array.Empty<float>(),
            output.PredictedBoundingBoxes ?? Array.Empty<float>()
        );
    }

    private static MlnetModelKind GetMlnetModelKind(ModelOptionResponse selectedModel)
    {
        if (
            selectedModel.AnnotationType.Equals("fully-annotated", StringComparison.OrdinalIgnoreCase)
            || selectedModel.Id.StartsWith("fullyannotated/", StringComparison.OrdinalIgnoreCase)
        )
        {
            return MlnetModelKind.FullyAnnotated;
        }

        return MlnetModelKind.HalfAnnotated;
    }

    private static string? GetContentType(string imagePath)
    {
        return Path.GetExtension(imagePath).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png",
            ".bmp" => "image/bmp",
            ".gif" => "image/gif",
            ".webp" => "image/webp",
            _ => null,
        };
    }

    private static IReadOnlyList<DetectionResponse> MapDetections(
        ModelOutputAdapter output,
        int imageWidth,
        int imageHeight
    )
    {
        var labels = output.PredictedLabels;
        var scores = output.Scores;
        var boxes = output.PredictedBoundingBoxes;
        var detectionCount = Math.Min(labels.Count, Math.Min(scores.Count, boxes.Count / 4));
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
        IReadOnlyList<float> boxes,
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

    private enum MlnetModelKind
    {
        HalfAnnotated,
        FullyAnnotated,
    }

    private sealed record ModelOutputAdapter(
        IReadOnlyList<string> PredictedLabels,
        IReadOnlyList<float> Scores,
        IReadOnlyList<float> PredictedBoundingBoxes
    );
}
