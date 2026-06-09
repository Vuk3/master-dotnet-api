using dotnet_api.Services;

namespace dotnet_api.Endpoints;

public static class ModelEndpoints
{
    public static WebApplication MapModelEndpoints(this WebApplication app)
    {
        app.MapGet("/models", (DotnetModelCatalogService modelCatalogService) =>
        {
            return Results.Ok(modelCatalogService.GetCatalog());
        })
        .WithName("GetModels")
        .WithOpenApi();

        return app;
    }
}
