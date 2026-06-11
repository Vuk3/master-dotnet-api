using System.Globalization;
using System.Text.Json;

namespace dotnet_api.Evaluation;

public static class CocoDatasetLoader
{
    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".webp",
    };

    public static CocoDataset Load(string annotationsPath, string imagesDirectory)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(annotationsPath));
        var root = document.RootElement;
        var imagePathLookup = BuildImagePathLookup(imagesDirectory);
        var categories = LoadCategories(root);
        var images = LoadImages(root, imagesDirectory, imagePathLookup);
        var annotations = LoadAnnotations(root, categories);
        var labels = annotations
            .Select(annotation => annotation.Label)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(label => label, StringComparer.OrdinalIgnoreCase)
            .ToList();

        return new CocoDataset(images, annotations, labels);
    }

    private static Dictionary<string, string> LoadCategories(JsonElement root)
    {
        var categories = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!root.TryGetProperty("categories", out var categoryElements))
        {
            return categories;
        }

        foreach (var category in categoryElements.EnumerateArray())
        {
            var id = ReadId(category, "id");
            var name = ReadString(category, "name");
            var supercategory = ReadString(category, "supercategory");

            if (id == "0" && supercategory.Equals("none", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!string.IsNullOrWhiteSpace(id) && !string.IsNullOrWhiteSpace(name))
            {
                categories[id] = name.Trim();
            }
        }

        return categories;
    }

    private static IReadOnlyList<CocoImage> LoadImages(
        JsonElement root,
        string imagesDirectory,
        IReadOnlyDictionary<string, string> imagePathLookup
    )
    {
        var images = new List<CocoImage>();

        if (!root.TryGetProperty("images", out var imageElements))
        {
            return images;
        }

        foreach (var image in imageElements.EnumerateArray())
        {
            var id = ReadId(image, "id");
            var fileName = ReadString(image, "file_name");

            if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(fileName))
            {
                continue;
            }

            images.Add(
                new CocoImage(
                    id,
                    fileName,
                    ReadInt(image, "width"),
                    ReadInt(image, "height"),
                    ResolveImagePath(imagesDirectory, fileName, imagePathLookup)
                )
            );
        }

        return images;
    }

    private static IReadOnlyList<CocoAnnotation> LoadAnnotations(
        JsonElement root,
        IReadOnlyDictionary<string, string> categories
    )
    {
        var annotations = new List<CocoAnnotation>();

        if (!root.TryGetProperty("annotations", out var annotationElements))
        {
            return annotations;
        }

        foreach (var annotation in annotationElements.EnumerateArray())
        {
            if (ReadBool(annotation, "iscrowd"))
            {
                continue;
            }

            var imageId = ReadId(annotation, "image_id");
            var categoryId = ReadId(annotation, "category_id");

            if (
                string.IsNullOrWhiteSpace(imageId)
                || string.IsNullOrWhiteSpace(categoryId)
                || !categories.TryGetValue(categoryId, out var label)
            )
            {
                continue;
            }

            if (
                !annotation.TryGetProperty("bbox", out var bboxElement)
                || bboxElement.ValueKind != JsonValueKind.Array
            )
            {
                continue;
            }

            var bbox = bboxElement.EnumerateArray().Select(ReadDouble).ToArray();
            if (bbox.Length < 4 || bbox[2] <= 0 || bbox[3] <= 0)
            {
                continue;
            }

            var box = DetectionBox.FromCoco(bbox[0], bbox[1], bbox[2], bbox[3]);
            annotations.Add(
                new CocoAnnotation(
                    imageId,
                    label,
                    box,
                    ReadDouble(annotation, "area", box.Area)
                )
            );
        }

        return annotations;
    }

    private static IReadOnlyDictionary<string, string> BuildImagePathLookup(string imagesDirectory)
    {
        if (!Directory.Exists(imagesDirectory))
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        return Directory
            .EnumerateFiles(imagesDirectory, "*", SearchOption.AllDirectories)
            .Where(path => ImageExtensions.Contains(Path.GetExtension(path)))
            .GroupBy(path => Path.GetFileName(path), StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.OrdinalIgnoreCase);
    }

    private static string? ResolveImagePath(
        string imagesDirectory,
        string fileName,
        IReadOnlyDictionary<string, string> imagePathLookup
    )
    {
        if (Path.IsPathFullyQualified(fileName) && File.Exists(fileName))
        {
            return fileName;
        }

        var relativePath = fileName
            .Replace('\\', Path.DirectorySeparatorChar)
            .Replace('/', Path.DirectorySeparatorChar);
        var directPath = Path.Combine(imagesDirectory, relativePath);
        if (File.Exists(directPath))
        {
            return directPath;
        }

        return imagePathLookup.TryGetValue(Path.GetFileName(fileName), out var resolvedPath)
            ? resolvedPath
            : null;
    }

    private static string ReadId(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var value))
        {
            return string.Empty;
        }

        return value.ValueKind switch
        {
            JsonValueKind.Number => value.TryGetInt64(out var number)
                ? number.ToString(CultureInfo.InvariantCulture)
                : value.GetDouble().ToString("0.########", CultureInfo.InvariantCulture),
            JsonValueKind.String => value.GetString() ?? string.Empty,
            _ => value.ToString(),
        };
    }

    private static string ReadString(JsonElement element, string propertyName)
    {
        return element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static int ReadInt(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var value))
        {
            return 0;
        }

        if (value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var number))
        {
            return number;
        }

        return value.ValueKind == JsonValueKind.String
            && int.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out number)
            ? number
            : 0;
    }

    private static bool ReadBool(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out var value))
        {
            return false;
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.Number => value.TryGetInt32(out var number) && number != 0,
            JsonValueKind.String => bool.TryParse(value.GetString(), out var booleanValue)
                ? booleanValue
                : value.GetString() == "1",
            _ => false,
        };
    }

    private static double ReadDouble(JsonElement value)
    {
        return value.ValueKind switch
        {
            JsonValueKind.Number => value.GetDouble(),
            JsonValueKind.String => double.TryParse(
                value.GetString(),
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out var number
            )
                ? number
                : 0,
            _ => 0,
        };
    }

    private static double ReadDouble(JsonElement element, string propertyName, double fallback)
    {
        return element.TryGetProperty(propertyName, out var value) ? ReadDouble(value) : fallback;
    }
}
