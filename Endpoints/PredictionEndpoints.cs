using Microsoft.AspNetCore.Mvc;

namespace dotnet_api.Endpoints;

public static class PredictionEndpoints
{
    public static WebApplication MapPredictionEndpoints(this WebApplication app)
    {
        app.MapPost("/predict", async ([FromForm] IFormFile file) =>
        {
            if (file == null || file.Length == 0)
            {
                return Results.BadRequest("No file uploaded.");
            }

            await Task.CompletedTask;

            return Results.Problem(
                title: "Predict unavailable",
                detail: "ML.NET model and generated prediction files were removed.",
                statusCode: StatusCodes.Status501NotImplemented
            );
        })
        .DisableAntiforgery()
        .WithName("Predict")
        .WithOpenApi();

        return app;
    }
}
