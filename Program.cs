var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenAnyIP(7146);
});


var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

var summaries = new[]
{
    "Freezing", "Bracing", "Chilly", "Cool", "Mild", "Warm", "Balmy", "Hot", "Sweltering", "Scorching"
};

// Postavljamo /predict kao POST endpoint
app.MapPost("/predict", async (IFormFile file) =>
{
    if (file == null || file.Length == 0)
    {
        return Results.BadRequest("No file uploaded.");
    }

    // Dummy response for testing
    return Results.Ok(new { message = "Prediction result from .NET API", fileName = file.FileName });
})
.WithName("Predict")
.WithOpenApi();

// Postavljamo /health kao GET endpoint
app.MapGet("/health", () =>
{
    return Results.Ok("Ok from DOTNET");
})
.WithName("Health")
.WithOpenApi();

app.Run();

record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
{
    public int TemperatureF => 32 + (int)(TemperatureC / 0.5556);
}
