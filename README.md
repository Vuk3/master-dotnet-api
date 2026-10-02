# Object Detection: ML.NET Service

The .NET half of my master's thesis, a system that runs the same image through a YOLOv8m model in
Python and an ML.NET model in .NET and compares the two. This is an ASP.NET Core minimal API that
runs an ML.NET object detection model over an uploaded image and answers in the same format as
the [Python service](https://github.com/Vuk3/master-python-api), so the
[gateway](https://github.com/Vuk3/master-gateway-api) and the frontend can treat the two as
interchangeable.

The full write-up, with the results: [vukcvetkovic.com/projects/object-detection](https://vukcvetkovic.com/projects/object-detection/)

## The models

Two models built with ML.NET Model Builder, which trains them through TorchSharp, on the same
2,911 images and six classes as the YOLOv8m models. Here too they differ only in the annotations.

| Model id | Annotations | Precision | Recall | F1 | mAP@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `halfannotated/halfannotatedmodel` | 8,813 | 0.669 | 0.645 | 0.657 | 0.580 |
| `fullyannotated/fullyannotatedmodel` | 17,942 | 0.862 | 0.709 | 0.778 | 0.748 |

The fully annotated model is the default. `runs/` holds the evaluation of each model on the
validation set: the precision, recall, F1 and PR curves, the confusion matrices, and the raw
predictions and matches behind them.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | The service's health check |
| `GET` | `/models` | Every `.mlnet` model under `Models/`, and the default |
| `POST` | `/predict` | The detections for one image |

`predict` takes `multipart/form-data`: `file` is the image, and `model` is an optional id from
`/models`. In the Development environment, Swagger UI is at `/swagger`.

## Evaluating a model

The same executable evaluates a model against a folder of images with COCO annotations:

```bash
dotnet run -- evaluate --images <valid-folder> --annotations <annotations.coco> \
  --model fullyannotated/fullyannotatedmodel --out runs/fully-annotated/mlnet_eval_valid
```

It runs the images in isolated worker processes, ten per worker by default, with a timeout and a
retry for each, so a native failure inside TorchSharp costs one batch rather than the whole run.
`dotnet run -- evaluate --help` lists every option.

## Running it

The .NET 8 SDK:

```bash
dotnet run
```

The service listens on `http://localhost:7146`, where the gateway expects it, and loads the
default model in the background at startup, so the first request does not wait for it.

The project references `TorchSharp-cuda-windows`, the CUDA build for Windows. `TorchSharp-cpu` is
the CPU build of the same version, for macOS, Linux or a machine without CUDA.
