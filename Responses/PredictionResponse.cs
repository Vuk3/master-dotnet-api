using System.Text.Json.Serialization;

namespace dotnet_api.Responses;

public record BoundingBoxResponse(
    [property: JsonPropertyName("x1")]
    float X1,
    [property: JsonPropertyName("y1")]
    float Y1,
    [property: JsonPropertyName("x2")]
    float X2,
    [property: JsonPropertyName("y2")]
    float Y2
);

public record DetectionResponse(
    [property: JsonPropertyName("label")]
    string Label,
    [property: JsonPropertyName("score")]
    float Score,
    [property: JsonPropertyName("box")]
    BoundingBoxResponse Box
);

public record PredictionResponse(
    [property: JsonPropertyName("model")]
    string Model,
    [property: JsonPropertyName("imageWidth")]
    int ImageWidth,
    [property: JsonPropertyName("imageHeight")]
    int ImageHeight,
    [property: JsonPropertyName("detections")]
    IReadOnlyList<DetectionResponse> Detections,
    [property: JsonPropertyName("fileName")]
    string? FileName,
    [property: JsonPropertyName("contentType")]
    string? ContentType
);
