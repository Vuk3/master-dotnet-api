namespace dotnet_api.Endpoints;

public static class HealthEndpoints
{
    public static WebApplication MapHealthEndpoints(this WebApplication app)
    {
        app.MapGet("/health", () =>
        {
            return Results.Ok("Ok from DOTNET");
        })
        .WithName("Health")
        .WithOpenApi();

        return app;
    }
}
