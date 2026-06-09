using dotnet_api.Responses;
using System.Text.RegularExpressions;

namespace dotnet_api.Services;

public class DotnetModelCatalogService
{
    private const string ModelExtension = ".mlnet";
    private static readonly Regex AnnotationWordBoundaryPattern = new("([a-z0-9])([A-Z])");
    private readonly Lazy<IReadOnlyList<ModelOptionResponse>> models;

    public DotnetModelCatalogService()
    {
        models = new Lazy<IReadOnlyList<ModelOptionResponse>>(LoadModels, true);
    }

    public ModelCatalogResponse GetCatalog()
    {
        var availableModels = models.Value;
        return new ModelCatalogResponse(
            availableModels,
            availableModels.FirstOrDefault(model => model.IsDefault)?.Id
        );
    }

    public ModelOptionResponse? GetModel(string? modelId)
    {
        var availableModels = models.Value;

        if (string.IsNullOrWhiteSpace(modelId))
        {
            return availableModels.FirstOrDefault(model => model.IsDefault)
                ?? availableModels.FirstOrDefault();
        }

        return availableModels.FirstOrDefault(
            model => model.Id.Equals(modelId, StringComparison.OrdinalIgnoreCase)
        );
    }

    public bool IsKnownModel(string? modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return true;
        }

        return GetModel(modelId) is not null;
    }

    private static IReadOnlyList<ModelOptionResponse> LoadModels()
    {
        var modelsRoot = Path.Combine(AppContext.BaseDirectory, "Models");
        if (!Directory.Exists(modelsRoot))
        {
            return Array.Empty<ModelOptionResponse>();
        }

        var modelPaths = Directory
            .EnumerateFiles(modelsRoot, $"*{ModelExtension}", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (modelPaths.Count == 0)
        {
            return Array.Empty<ModelOptionResponse>();
        }

        var defaultModelPath = modelPaths[0];

        return modelPaths
            .Select(path => CreateModelSummary(modelsRoot, path, path == defaultModelPath))
            .ToList();
    }

    private static ModelOptionResponse CreateModelSummary(
        string modelsRoot,
        string modelPath,
        bool isDefault
    )
    {
        var annotationType = GetAnnotationType(modelsRoot, modelPath);

        return new ModelOptionResponse(
            GetModelId(modelsRoot, modelPath),
            "ML.NET Object Detection",
            "ML.NET Model Builder",
            annotationType,
            isDefault
        );
    }

    private static string GetModelId(string modelsRoot, string modelPath)
    {
        var relativePath = Path.GetRelativePath(modelsRoot, modelPath);
        return Path.ChangeExtension(relativePath, null)
            .Replace('\\', '/')
            .ToLowerInvariant();
    }

    private static string GetAnnotationType(string modelsRoot, string modelPath)
    {
        var relativeDirectory = Path.GetRelativePath(
            modelsRoot,
            Path.GetDirectoryName(modelPath) ?? modelsRoot
        );

        if (string.IsNullOrWhiteSpace(relativeDirectory) || relativeDirectory == ".")
        {
            return "default";
        }

        var firstSegment = relativeDirectory
            .Replace('\\', '/')
            .Split('/', StringSplitOptions.RemoveEmptyEntries)
            .FirstOrDefault();

        return string.IsNullOrWhiteSpace(firstSegment) ? "default" : NormalizeAnnotationType(firstSegment);
    }

    private static string NormalizeAnnotationType(string value)
    {
        var normalizedValue = value.Replace('_', '-').Replace(' ', '-');
        normalizedValue = AnnotationWordBoundaryPattern.Replace(normalizedValue, "$1-$2");
        return normalizedValue.ToLowerInvariant();
    }
}
