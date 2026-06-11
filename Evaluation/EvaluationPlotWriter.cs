using System.Globalization;
using SkiaSharp;

namespace dotnet_api.Evaluation;

public static class EvaluationPlotWriter
{
    private const string AllClassesLabel = "all";
    private const string BackgroundLabel = "background";
    private static readonly SKColor[] Palette =
    [
        new(31, 119, 180),
        new(255, 127, 14),
        new(44, 160, 44),
        new(214, 39, 40),
        new(148, 103, 189),
        new(140, 86, 75),
        new(227, 119, 194),
        new(127, 127, 127),
        new(188, 189, 34),
        new(23, 190, 207),
    ];

    public static void Write(EvaluationRun run, string outputDirectory)
    {
        WriteConfidenceCurve(
            run,
            Path.Combine(outputDirectory, "BoxF1_curve.png"),
            "F1-Confidence Curve",
            "F1",
            metric => metric.F1,
            "0.00"
        );
        WriteConfidenceCurve(
            run,
            Path.Combine(outputDirectory, "BoxP_curve.png"),
            "Precision-Confidence Curve",
            "Precision",
            metric => metric.Precision,
            "0.00"
        );
        WriteConfidenceCurve(
            run,
            Path.Combine(outputDirectory, "BoxR_curve.png"),
            "Recall-Confidence Curve",
            "Recall",
            metric => metric.Recall,
            "0.00"
        );
        WritePrecisionRecallCurve(run, Path.Combine(outputDirectory, "BoxPR_curve.png"));
        WriteConfusionMatrix(
            run.ConfusionMatrix,
            Path.Combine(outputDirectory, "confusion_matrix.png"),
            "Confusion Matrix",
            false
        );
        WriteConfusionMatrix(
            run.ConfusionMatrix,
            Path.Combine(outputDirectory, "confusion_matrix_normalized.png"),
            "Confusion Matrix Normalized",
            true
        );
        WriteLabels(run, Path.Combine(outputDirectory, "labels.jpg"));
    }

    private static void WriteConfidenceCurve(
        EvaluationRun run,
        string outputPath,
        string title,
        string yLabel,
        Func<MetricPoint, double> valueSelector,
        string valueFormat
    )
    {
        var labels = GetLabels(run);
        var series = new List<LineSeries>();

        for (var index = 0; index < labels.Count; index++)
        {
            var label = labels[index];
            var points = run.Metrics
                .Where(metric => metric.Label.Equals(label, StringComparison.OrdinalIgnoreCase))
                .OrderBy(metric => metric.Confidence)
                .Select(metric => new DataPoint(metric.Confidence, valueSelector(metric)))
                .ToList();

            series.Add(
                new LineSeries(
                    label,
                    label,
                    points,
                    Palette[index % Palette.Length],
                    3
                )
            );
        }

        var allMetrics = run.Metrics
            .Where(metric => metric.Label == AllClassesLabel)
            .OrderBy(metric => metric.Confidence)
            .ToList();
        var best = allMetrics
            .OrderByDescending(valueSelector)
            .ThenBy(metric => metric.Confidence)
            .FirstOrDefault();
        var allLegend = best is null
            ? "all classes"
            : $"all classes {valueSelector(best).ToString(valueFormat, CultureInfo.InvariantCulture)} at {best.Confidence.ToString("0.###", CultureInfo.InvariantCulture)}";

        series.Add(
            new LineSeries(
                AllClassesLabel,
                allLegend,
                allMetrics
                    .Select(metric => new DataPoint(metric.Confidence, valueSelector(metric)))
                    .ToList(),
                SKColors.Blue,
                9
            )
        );

        DrawLinePlot(
            outputPath,
            title,
            "Confidence",
            yLabel,
            series,
            0,
            1,
            0,
            1
        );
    }

    private static void WritePrecisionRecallCurve(EvaluationRun run, string outputPath)
    {
        var labels = GetLabels(run);
        var series = new List<LineSeries>();
        var classCurves = new List<IReadOnlyList<DataPoint>>();

        for (var index = 0; index < labels.Count; index++)
        {
            var label = labels[index];
            var averagePrecision = run.AveragePrecisions.FirstOrDefault(
                result => result.Label.Equals(label, StringComparison.OrdinalIgnoreCase)
            );

            if (averagePrecision is null)
            {
                continue;
            }

            var curvePoints = BuildPrecisionRecallPlotCurve(averagePrecision.Curve);
            classCurves.Add(curvePoints);
            series.Add(
                new LineSeries(
                    label,
                    $"{label} {averagePrecision.AveragePrecision.ToString("0.000", CultureInfo.InvariantCulture)}",
                    curvePoints,
                    Palette[index % Palette.Length],
                    3
                )
            );
        }

        var allPoints = BuildMeanPrecisionRecallCurve(classCurves);
        series.Add(
            new LineSeries(
                AllClassesLabel,
                $"all classes {run.Summary.Map50.ToString("0.000", CultureInfo.InvariantCulture)} mAP@0.5",
                allPoints,
                SKColors.Blue,
                9
            )
        );

        DrawLinePlot(
            outputPath,
            "Precision-Recall Curve",
            "Recall",
            "Precision",
            series,
            0,
            1,
            0,
            1
        );
    }

    private static IReadOnlyList<DataPoint> BuildPrecisionRecallPlotCurve(
        IReadOnlyList<PrecisionRecallPoint> rawCurve
    )
    {
        if (rawCurve.Count == 0)
        {
            return [new DataPoint(0, 0), new DataPoint(1, 0)];
        }

        var points = new List<DataPoint>
        {
            new(0, 1),
        };
        points.AddRange(
            rawCurve
                .OrderBy(point => point.Recall)
                .Select(point => new DataPoint(
                    Math.Clamp(point.Recall, 0, 1),
                    Math.Clamp(point.Precision, 0, 1)
                ))
        );
        points.Add(new DataPoint(1, 0));

        var envelope = points
            .GroupBy(point => point.X)
            .Select(group => new DataPoint(group.Key, group.Max(point => point.Y)))
            .OrderBy(point => point.X)
            .ToList();

        for (var index = envelope.Count - 2; index >= 0; index--)
        {
            envelope[index] = envelope[index] with
            {
                Y = Math.Max(envelope[index].Y, envelope[index + 1].Y),
            };
        }

        return envelope;
    }

    private static IReadOnlyList<DataPoint> BuildMeanPrecisionRecallCurve(
        IReadOnlyList<IReadOnlyList<DataPoint>> classCurves
    )
    {
        if (classCurves.Count == 0)
        {
            return [new DataPoint(0, 0), new DataPoint(1, 0)];
        }

        var points = new List<DataPoint>();
        for (var index = 0; index <= 100; index++)
        {
            var recall = index / 100.0;
            var precision = classCurves.Average(curve => InterpolatePrecision(curve, recall));
            points.Add(new DataPoint(recall, precision));
        }

        return points;
    }

    private static double InterpolatePrecision(IReadOnlyList<DataPoint> curve, double recall)
    {
        if (curve.Count == 0)
        {
            return 0;
        }

        if (recall <= curve[0].X)
        {
            return curve[0].Y;
        }

        for (var index = 1; index < curve.Count; index++)
        {
            var right = curve[index];
            if (recall > right.X)
            {
                continue;
            }

            var left = curve[index - 1];
            if (Math.Abs(right.X - left.X) < double.Epsilon)
            {
                return Math.Max(left.Y, right.Y);
            }

            var ratio = (recall - left.X) / (right.X - left.X);
            return left.Y + ratio * (right.Y - left.Y);
        }

        return curve[^1].Y;
    }

    private static void DrawLinePlot(
        string outputPath,
        string title,
        string xLabel,
        string yLabel,
        IReadOnlyList<LineSeries> series,
        double xMin,
        double xMax,
        double yMin,
        double yMax
    )
    {
        const int width = 2250;
        const int height = 1500;
        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var plot = new SKRect(150, 120, 1410, 1320);
        var legend = new SKRect(1480, 100, 2190, 100 + Math.Max(1, series.Count) * 62 + 36);

        DrawTitle(canvas, title, width / 2f, 65);
        DrawAxes(canvas, plot, xLabel, yLabel);
        DrawUnitTicks(canvas, plot);

        canvas.Save();
        canvas.ClipRect(plot);
        foreach (var item in series)
        {
            DrawSeries(canvas, plot, item, xMin, xMax, yMin, yMax);
        }
        canvas.Restore();

        DrawLegend(canvas, legend, series);
        SaveBitmap(bitmap, outputPath, SKEncodedImageFormat.Png);
    }

    private static void WriteConfusionMatrix(
        ConfusionMatrixResult matrix,
        string outputPath,
        string title,
        bool normalized
    )
    {
        const int width = 3000;
        const int height = 2250;
        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var labels = matrix.Labels;
        var size = labels.Count;
        var plotSize = Math.Min(1650, 220 * size);
        var plot = new SKRect(710, 155, 710 + plotSize, 155 + plotSize);
        var cellSize = plot.Width / size;
        var maxValue = normalized
            ? 1
            : Enumerable.Range(0, size)
                .SelectMany(row => Enumerable.Range(0, size).Select(column => matrix.Counts[row, column]))
                .DefaultIfEmpty(0)
                .Max();

        DrawTitle(canvas, title, width / 2f, 62);

        using var borderPaint = new SKPaint
        {
            Color = SKColors.White,
            IsAntialias = true,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 2,
        };
        using var labelPaint = TextPaint(48, SKColors.Black, SKTextAlign.Right);
        using var valuePaint = TextPaint(38, SKColors.Black, SKTextAlign.Center);

        for (var row = 0; row < size; row++)
        {
            for (var column = 0; column < size; column++)
            {
                var value = normalized ? matrix.Normalized[row, column] : matrix.Counts[row, column];
                var rect = new SKRect(
                    plot.Left + column * cellSize,
                    plot.Top + row * cellSize,
                    plot.Left + (column + 1) * cellSize,
                    plot.Top + (row + 1) * cellSize
                );
                using var fillPaint = new SKPaint
                {
                    Color = HeatColor(maxValue <= 0 ? 0 : value / maxValue),
                    IsAntialias = true,
                    Style = SKPaintStyle.Fill,
                };
                canvas.DrawRect(rect, fillPaint);
                canvas.DrawRect(rect, borderPaint);

                if (value <= 0)
                {
                    continue;
                }

                var text = normalized
                    ? value.ToString("0.00", CultureInfo.InvariantCulture)
                    : ((int)value).ToString(CultureInfo.InvariantCulture);
                valuePaint.Color = value / Math.Max(maxValue, 1e-9) > 0.48 ? SKColors.White : SKColors.Black;
                canvas.DrawText(text, rect.MidX, rect.MidY + valuePaint.TextSize / 3, valuePaint);
            }
        }

        for (var index = 0; index < size; index++)
        {
            var center = plot.Top + index * cellSize + cellSize / 2;
            canvas.DrawText(labels[index], plot.Left - 28, center + labelPaint.TextSize / 3, labelPaint);

            canvas.Save();
            canvas.Translate(plot.Left + index * cellSize + cellSize / 2 + 12, plot.Bottom + 36);
            canvas.RotateDegrees(90);
            using var xLabelPaint = TextPaint(48, SKColors.Black, SKTextAlign.Left);
            canvas.DrawText(labels[index], 0, 0, xLabelPaint);
            canvas.Restore();
        }

        using var axisPaint = TextPaint(38, SKColors.Black, SKTextAlign.Center);
        canvas.DrawText("True", plot.MidX, Math.Min(height - 85, plot.Bottom + 380), axisPaint);
        canvas.Save();
        canvas.Translate(300, plot.MidY);
        canvas.RotateDegrees(-90);
        canvas.DrawText("Predicted", 0, 0, axisPaint);
        canvas.Restore();

        DrawColorBar(canvas, new SKRect(plot.Right + 135, plot.Top, plot.Right + 210, plot.Bottom), normalized, maxValue);
        SaveBitmap(bitmap, outputPath, SKEncodedImageFormat.Png);
    }

    private static void WriteLabels(EvaluationRun run, string outputPath)
    {
        const int width = 1600;
        const int height = 1600;
        using var bitmap = new SKBitmap(width, height);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        var labels = GetLabels(run);
        var boxes = run.Images
            .SelectMany(image => image.GroundTruth.Select(annotation => new NormalizedBox(
                annotation.Label,
                image.Width <= 0 ? 0 : annotation.Box.CenterX / image.Width,
                image.Height <= 0 ? 0 : annotation.Box.CenterY / image.Height,
                image.Width <= 0 ? 0 : annotation.Box.Width / image.Width,
                image.Height <= 0 ? 0 : annotation.Box.Height / image.Height
            )))
            .Where(box => box.Width > 0 && box.Height > 0)
            .ToList();

        DrawLabelBars(canvas, new SKRect(120, 80, 760, 580), labels, boxes);
        DrawNormalizedBoxes(canvas, new SKRect(940, 95, 1450, 580), boxes);
        DrawHeatmapPanel(canvas, new SKRect(130, 880, 760, 1490), boxes, box => box.CenterX, box => box.CenterY, "x", "y");
        DrawHeatmapPanel(canvas, new SKRect(920, 880, 1530, 1490), boxes, box => box.Width, box => box.Height, "width", "height");

        SaveBitmap(bitmap, outputPath, SKEncodedImageFormat.Jpeg);
    }

    private static void DrawLabelBars(
        SKCanvas canvas,
        SKRect plot,
        IReadOnlyList<string> labels,
        IReadOnlyList<NormalizedBox> boxes
    )
    {
        var counts = labels
            .Select(label => boxes.Count(box => box.Label.Equals(label, StringComparison.OrdinalIgnoreCase)))
            .ToList();
        var maxCount = Math.Max(1, counts.DefaultIfEmpty(0).Max());

        DrawSimpleAxes(canvas, plot, "instances", null);

        using var textPaint = TextPaint(34, SKColors.Black, SKTextAlign.Center);
        using var labelPaint = TextPaint(31, SKColors.Black, SKTextAlign.Left);
        var barGap = 18f;
        var barWidth = labels.Count == 0 ? 0 : (plot.Width - barGap * (labels.Count + 1)) / labels.Count;

        for (var index = 0; index < labels.Count; index++)
        {
            var barHeight = (float)(counts[index] / (double)maxCount * plot.Height);
            var left = plot.Left + barGap + index * (barWidth + barGap);
            var rect = new SKRect(left, plot.Bottom - barHeight, left + barWidth, plot.Bottom);
            using var fillPaint = new SKPaint
            {
                Color = Palette[index % Palette.Length],
                IsAntialias = true,
                Style = SKPaintStyle.Fill,
            };
            canvas.DrawRect(rect, fillPaint);
            canvas.DrawText(counts[index].ToString(), rect.MidX, rect.Top - 12, textPaint);

            canvas.Save();
            canvas.Translate(rect.MidX - 4, plot.Bottom + 28);
            canvas.RotateDegrees(90);
            canvas.DrawText(labels[index], 0, 0, labelPaint);
            canvas.Restore();
        }
    }

    private static void DrawNormalizedBoxes(
        SKCanvas canvas,
        SKRect plot,
        IReadOnlyList<NormalizedBox> boxes
    )
    {
        DrawSimpleAxes(canvas, plot, null, null);
        using var strokePaint = new SKPaint
        {
            Color = new SKColor(22, 34, 90, 80),
            IsAntialias = true,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 1,
        };

        foreach (var box in boxes)
        {
            var halfWidth = (float)(box.Width * plot.Width / 2);
            var halfHeight = (float)(box.Height * plot.Height / 2);
            var centerX = plot.MidX;
            var centerY = plot.MidY;
            canvas.DrawRect(
                new SKRect(
                    centerX - halfWidth,
                    centerY - halfHeight,
                    centerX + halfWidth,
                    centerY + halfHeight
                ),
                strokePaint
            );
        }
    }

    private static void DrawHeatmapPanel(
        SKCanvas canvas,
        SKRect plot,
        IReadOnlyList<NormalizedBox> boxes,
        Func<NormalizedBox, double> xSelector,
        Func<NormalizedBox, double> ySelector,
        string xLabel,
        string yLabel
    )
    {
        const int bins = 48;
        var histogram = new int[bins, bins];
        var maxBin = 0;

        foreach (var box in boxes)
        {
            var x = Math.Clamp(xSelector(box), 0, 0.999999);
            var y = Math.Clamp(ySelector(box), 0, 0.999999);
            var xIndex = (int)(x * bins);
            var yIndex = (int)(y * bins);
            histogram[xIndex, yIndex]++;
            maxBin = Math.Max(maxBin, histogram[xIndex, yIndex]);
        }

        DrawSimpleAxes(canvas, plot, yLabel, xLabel);
        var cellWidth = plot.Width / bins;
        var cellHeight = plot.Height / bins;

        for (var x = 0; x < bins; x++)
        {
            for (var y = 0; y < bins; y++)
            {
                var count = histogram[x, y];
                if (count == 0)
                {
                    continue;
                }

                var intensity = Math.Sqrt(count / (double)Math.Max(1, maxBin));
                using var fillPaint = new SKPaint
                {
                    Color = new SKColor(44, 80, 255, (byte)(35 + intensity * 185)),
                    IsAntialias = false,
                    Style = SKPaintStyle.Fill,
                };
                canvas.DrawRect(
                    new SKRect(
                        plot.Left + x * cellWidth,
                        plot.Bottom - (y + 1) * cellHeight,
                        plot.Left + (x + 1) * cellWidth,
                        plot.Bottom - y * cellHeight
                    ),
                    fillPaint
                );
            }
        }
    }

    private static void DrawSeries(
        SKCanvas canvas,
        SKRect plot,
        LineSeries series,
        double xMin,
        double xMax,
        double yMin,
        double yMax
    )
    {
        if (series.Points.Count == 0)
        {
            return;
        }

        using var path = new SKPath();
        var hasStarted = false;
        foreach (var point in series.Points)
        {
            var x = Map(point.X, xMin, xMax, plot.Left, plot.Right);
            var y = Map(point.Y, yMin, yMax, plot.Bottom, plot.Top);

            if (!hasStarted)
            {
                path.MoveTo(x, y);
                hasStarted = true;
            }
            else
            {
                path.LineTo(x, y);
            }
        }

        using var paint = new SKPaint
        {
            Color = series.Color,
            IsAntialias = true,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = series.StrokeWidth,
        };
        canvas.DrawPath(path, paint);
    }

    private static void DrawAxes(SKCanvas canvas, SKRect plot, string xLabel, string yLabel)
    {
        using var axisPaint = new SKPaint
        {
            Color = SKColors.Black,
            IsAntialias = true,
            StrokeWidth = 3,
            Style = SKPaintStyle.Stroke,
        };
        canvas.DrawLine(plot.Left, plot.Bottom, plot.Right, plot.Bottom, axisPaint);
        canvas.DrawLine(plot.Left, plot.Top, plot.Left, plot.Bottom, axisPaint);
        canvas.DrawLine(plot.Right, plot.Top, plot.Right, plot.Bottom, axisPaint);
        canvas.DrawLine(plot.Left, plot.Top, plot.Right, plot.Top, axisPaint);

        using var labelPaint = TextPaint(42, SKColors.Black, SKTextAlign.Center);
        canvas.DrawText(xLabel, plot.MidX, plot.Bottom + 95, labelPaint);
        canvas.Save();
        canvas.Translate(55, plot.MidY);
        canvas.RotateDegrees(-90);
        canvas.DrawText(yLabel, 0, 0, labelPaint);
        canvas.Restore();
    }

    private static void DrawSimpleAxes(SKCanvas canvas, SKRect plot, string? yLabel, string? xLabel)
    {
        using var axisPaint = new SKPaint
        {
            Color = SKColors.Black,
            IsAntialias = true,
            StrokeWidth = 2,
            Style = SKPaintStyle.Stroke,
        };
        canvas.DrawRect(plot, axisPaint);

        using var labelPaint = TextPaint(31, SKColors.Black, SKTextAlign.Center);
        if (!string.IsNullOrWhiteSpace(xLabel))
        {
            canvas.DrawText(xLabel, plot.MidX, plot.Bottom + 65, labelPaint);
        }

        if (!string.IsNullOrWhiteSpace(yLabel))
        {
            canvas.Save();
            canvas.Translate(plot.Left - 82, plot.MidY);
            canvas.RotateDegrees(-90);
            canvas.DrawText(yLabel, 0, 0, labelPaint);
            canvas.Restore();
        }
    }

    private static void DrawUnitTicks(SKCanvas canvas, SKRect plot)
    {
        using var gridPaint = new SKPaint
        {
            Color = new SKColor(225, 225, 225),
            IsAntialias = true,
            StrokeWidth = 1,
            Style = SKPaintStyle.Stroke,
        };
        using var tickPaint = new SKPaint
        {
            Color = SKColors.Black,
            IsAntialias = true,
            StrokeWidth = 2,
            Style = SKPaintStyle.Stroke,
        };
        using var textPaint = TextPaint(34, SKColors.Black, SKTextAlign.Center);

        for (var index = 0; index <= 5; index++)
        {
            var value = index / 5.0;
            var x = Map(value, 0, 1, plot.Left, plot.Right);
            var y = Map(value, 0, 1, plot.Bottom, plot.Top);

            canvas.DrawLine(x, plot.Top, x, plot.Bottom, gridPaint);
            canvas.DrawLine(plot.Left, y, plot.Right, y, gridPaint);
            canvas.DrawLine(x, plot.Bottom, x, plot.Bottom + 12, tickPaint);
            canvas.DrawLine(plot.Left - 12, y, plot.Left, y, tickPaint);
            canvas.DrawText(value.ToString("0.0", CultureInfo.InvariantCulture), x, plot.Bottom + 50, textPaint);
            canvas.DrawText(
                value.ToString("0.0", CultureInfo.InvariantCulture),
                plot.Left - 48,
                y + textPaint.TextSize / 3,
                textPaint
            );
        }
    }

    private static void DrawLegend(SKCanvas canvas, SKRect legend, IReadOnlyList<LineSeries> series)
    {
        using var fillPaint = new SKPaint
        {
            Color = SKColors.White,
            IsAntialias = true,
            Style = SKPaintStyle.Fill,
        };
        using var borderPaint = new SKPaint
        {
            Color = new SKColor(205, 205, 205),
            IsAntialias = true,
            StrokeWidth = 2,
            Style = SKPaintStyle.Stroke,
        };
        canvas.DrawRoundRect(legend, 6, 6, fillPaint);
        canvas.DrawRoundRect(legend, 6, 6, borderPaint);

        using var textPaint = TextPaint(34, SKColors.Black, SKTextAlign.Left);
        var y = legend.Top + 50;
        foreach (var item in series)
        {
            using var linePaint = new SKPaint
            {
                Color = item.Color,
                IsAntialias = true,
                StrokeWidth = item.StrokeWidth,
                Style = SKPaintStyle.Stroke,
            };
            canvas.DrawLine(legend.Left + 34, y - 10, legend.Left + 104, y - 10, linePaint);
            canvas.DrawText(item.Legend, legend.Left + 130, y, textPaint);
            y += 62;
        }
    }

    private static void DrawColorBar(
        SKCanvas canvas,
        SKRect bar,
        bool normalized,
        double maxValue
    )
    {
        for (var y = 0; y < bar.Height; y++)
        {
            var ratio = 1 - y / bar.Height;
            using var paint = new SKPaint
            {
                Color = HeatColor(ratio),
                Style = SKPaintStyle.Fill,
            };
            canvas.DrawRect(new SKRect(bar.Left, bar.Top + y, bar.Right, bar.Top + y + 1), paint);
        }

        using var borderPaint = new SKPaint
        {
            Color = SKColors.White,
            StrokeWidth = 1,
            Style = SKPaintStyle.Stroke,
        };
        canvas.DrawRect(bar, borderPaint);

        using var textPaint = TextPaint(26, SKColors.Black, SKTextAlign.Left);
        for (var index = 0; index <= 5; index++)
        {
            var ratio = index / 5.0;
            var value = normalized ? ratio : ratio * maxValue;
            var y = Map(ratio, 0, 1, bar.Bottom, bar.Top);
            canvas.DrawText(
                normalized
                    ? value.ToString("0.0", CultureInfo.InvariantCulture)
                    : value.ToString("0", CultureInfo.InvariantCulture),
                bar.Right + 18,
                y + textPaint.TextSize / 3,
                textPaint
            );
        }
    }

    private static void DrawTitle(SKCanvas canvas, string title, float x, float y)
    {
        using var paint = TextPaint(48, SKColors.Black, SKTextAlign.Center);
        canvas.DrawText(title, x, y, paint);
    }

    private static SKPaint TextPaint(float size, SKColor color, SKTextAlign align)
    {
        return new SKPaint
        {
            Color = color,
            IsAntialias = true,
            TextSize = size,
            TextAlign = align,
        };
    }

    private static SKColor HeatColor(double ratio)
    {
        ratio = Math.Clamp(ratio, 0, 1);
        var inverse = 1 - ratio;
        var r = (byte)(247 * inverse + 8 * ratio);
        var g = (byte)(251 * inverse + 72 * ratio);
        var b = (byte)(255 * inverse + 140 * ratio);
        return new SKColor(r, g, b);
    }

    private static float Map(double value, double sourceMin, double sourceMax, float targetMin, float targetMax)
    {
        if (Math.Abs(sourceMax - sourceMin) < double.Epsilon)
        {
            return targetMin;
        }

        var ratio = (value - sourceMin) / (sourceMax - sourceMin);
        return (float)(targetMin + ratio * (targetMax - targetMin));
    }

    private static IReadOnlyList<string> GetLabels(EvaluationRun run)
    {
        return run.ConfusionMatrix.Labels
            .Where(label => !label.Equals(BackgroundLabel, StringComparison.OrdinalIgnoreCase))
            .ToList();
    }

    private static void SaveBitmap(SKBitmap bitmap, string outputPath, SKEncodedImageFormat format)
    {
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(format, 95);
        using var stream = File.Open(outputPath, FileMode.Create, FileAccess.Write);
        data.SaveTo(stream);
    }

    private sealed record DataPoint(double X, double Y);

    private sealed record LineSeries(
        string Label,
        string Legend,
        IReadOnlyList<DataPoint> Points,
        SKColor Color,
        float StrokeWidth
    );

    private sealed record NormalizedBox(
        string Label,
        double CenterX,
        double CenterY,
        double Width,
        double Height
    );
}
