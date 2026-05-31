using Microsoft.AspNetCore.Mvc;

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
app.UseDeveloperExceptionPage();

// Postavljamo /predict kao POST endpoint

app.MapPost("/predict", async ([FromForm] IFormFile file) =>
{
    if (file == null || file.Length == 0)
        return Results.BadRequest("No file uploaded.");

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

app.MapGet("/health", () =>
{
    return Results.Ok("Ok from DOTNET");
})
.WithName("Health")
.WithOpenApi();

app.Run();
