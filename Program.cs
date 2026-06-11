using dotnet_api.Evaluation;
using dotnet_api.Endpoints;
using dotnet_api.Services;
using Microsoft.Extensions.Logging.Abstractions;

if (
    EvaluationCommand.IsEvaluationCommand(args)
    || EvaluationCommand.IsPredictionWorkerCommand(args)
    || EvaluationCommand.IsPredictionBatchWorkerCommand(args)
)
{
    var modelCatalogService = new DotnetModelCatalogService();
    var evaluationPredictionService = new MlnetPredictionService(
        NullLogger<MlnetPredictionService>.Instance,
        modelCatalogService
    );
    Environment.ExitCode = EvaluationCommand.IsPredictionBatchWorkerCommand(args)
        ? EvaluationCommand.RunPredictionBatchWorker(evaluationPredictionService, args.Skip(1).ToArray())
        : EvaluationCommand.IsPredictionWorkerCommand(args)
            ? EvaluationCommand.RunPredictionWorker(evaluationPredictionService, args.Skip(1).ToArray())
            : EvaluationCommand.Run(evaluationPredictionService, args.Skip(1).ToArray());
    return;
}

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<DotnetModelCatalogService>();
builder.Services.AddSingleton<MlnetPredictionService>();

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

var predictionService = app.Services.GetRequiredService<MlnetPredictionService>();
_ = Task.Run(predictionService.WarmUp);

app.Run();
