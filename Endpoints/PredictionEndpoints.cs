using Microsoft.AspNetCore.Mvc;
using dotnet_api.Services;

namespace dotnet_api.Endpoints;

public static class PredictionEndpoints
{
    public static WebApplication MapPredictionEndpoints(this WebApplication app)
    {
        app.MapPost("/predict", async (
            [FromForm] IFormFile file,
            [FromForm] string? model,
            DotnetModelCatalogService modelCatalogService,
            HalfAnnotatedPredictionService predictionService
        ) =>
        {
            if (file == null || file.Length == 0)
            {
                return Results.BadRequest("No file uploaded.");
            }

            if (!modelCatalogService.IsKnownModel(model))
            {
                return Results.BadRequest("Requested .NET model was not found.");
            }

            var prediction = await predictionService.Predict(file, model);

            return Results.Ok(prediction);
        })
        .DisableAntiforgery()
        .WithName("Predict")
        .WithOpenApi();

        return app;
    }
}
