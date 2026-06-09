using dotnet_api.Endpoints;
using dotnet_api.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<DotnetModelCatalogService>();
builder.Services.AddSingleton<HalfAnnotatedPredictionService>();

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

app.MapHealthEndpoints();
app.MapModelEndpoints();
app.MapPredictionEndpoints();

var predictionService = app.Services.GetRequiredService<HalfAnnotatedPredictionService>();
_ = Task.Run(predictionService.WarmUp);

app.Run();
