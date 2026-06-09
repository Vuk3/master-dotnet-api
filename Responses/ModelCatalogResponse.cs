using System.Text.Json.Serialization;

namespace dotnet_api.Responses;

public record ModelOptionResponse(
    [property: JsonPropertyName("id")]
    string Id,
    [property: JsonPropertyName("name")]
    string Name,
    [property: JsonPropertyName("family")]
    string Family,
    [property: JsonPropertyName("annotationType")]
    string AnnotationType,
    [property: JsonPropertyName("isDefault")]
    bool IsDefault
);

public record ModelCatalogResponse(
    [property: JsonPropertyName("models")]
    IReadOnlyList<ModelOptionResponse> Models,
    [property: JsonPropertyName("defaultModelId")]
    string? DefaultModelId
);
