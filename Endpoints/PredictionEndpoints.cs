using Microsoft.AspNetCore.Mvc;
using dotnet_api.Services;

namespace dotnet_api.Endpoints;

public static class PredictionEndpoints
{
    public static WebApplication MapPredictionEndpoints(this WebApplication app)
    {
        app.MapPost("/predict", async (
            [FromForm] IFormFile file,
            HalfAnnotatedPredictionService predictionService
        ) =>
        {
            if (file == null || file.Length == 0)
            {
                return Results.BadRequest("No file uploaded.");
            }

            var prediction = await predictionService.Predict(file);

            return Results.Ok(prediction);
        })
        .DisableAntiforgery()
        .WithName("Predict")
        .WithOpenApi();

        return app;
    }
}
