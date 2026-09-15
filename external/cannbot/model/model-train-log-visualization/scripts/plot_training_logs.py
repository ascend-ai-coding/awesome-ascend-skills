#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


def _load_train_log_metrics_module():
    module_path = Path(__file__).resolve().parent / "train_log_metrics.py"
    if not module_path.exists():
        raise RuntimeError(f"missing module file: {module_path}")
    spec = importlib.util.spec_from_file_location("train_log_metrics", module_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_log_metrics"] = module
    spec.loader.exec_module(module)
    return module


def _normalize_metric(metric: str) -> str:
    alias_map = {
        "memory": "memory_gib",
        "mfu": "mfu_pct",
        "elapsed": "elapsed_time_per_step",
        "indexer": "indexer_loss",
        "indexer_loss": "indexer_loss",
    }
    return alias_map.get(metric, metric)


def _parse_metric_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [_normalize_metric(item.strip()) for item in raw.split(",") if item.strip()]


def _default_output_path(log_a: str, log_b: str | None, output_format: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_a = Path(log_a).stem
    if log_b:
        base_b = Path(log_b).stem
        filename = f"{base_a}_vs_{base_b}_{stamp}.{output_format}"
    else:
        filename = f"{base_a}_{stamp}.{output_format}"
    return Path.cwd() / filename


def _require_matplotlib(no_show: bool):
    try:
        import matplotlib

        if no_show:
            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for plotting. Please install it first: pip install matplotlib"
        ) from exc


def _apply_integer_xaxis(axis) -> None:
    """Force the x-axis (training step) to display integer ticks only.

    MaxNLocator(integer=True) prefers integer ticks but still falls back to
    fractional values when the data range is narrow (e.g. a single point). Pair
    it with a formatter that hides labels for any non-integer tick so the
    rendered step axis never shows decimals.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def _format_step(value: float, _pos: int) -> str:
        rounded = round(value)
        if abs(value - rounded) > 1e-6:
            return ""
        return f"{int(rounded)}"

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_step))


def _metric_label(metric_key: str) -> str:
    labels = {
        "loss": "loss",
        "grad_norm": "grad_norm",
        "memory_gib": "memory (GB)",
        "memory_pct": "memory (%)",
        "tps": "tps",
        "tflops": "tflops",
        "mfu_pct": "mfu (%)",
        "elapsed_time_per_step": "elapsed_time_per_step (s)",
        "indexer_loss": "indexer loss",
        "loss_abs_error": "loss abs error",
        "loss_rel_error": "loss relative error",
        "grad_norm_abs_error": "grad_norm abs error",
        "grad_norm_rel_error": "grad_norm relative error",
    }
    return labels.get(metric_key, metric_key)


def _compute_error_stats(errors: list[float]) -> dict[str, float]:
    """NaN-aware mean/mse/min/max. All-NaN input returns NaN, not 0.0, to avoid
    masking missing-metric steps as a perfect match."""
    valid = [e for e in errors if not math.isnan(e)]
    if not valid:
        return {"mean": float("nan"), "mse": float("nan"), "min": float("nan"), "max": float("nan")}

    mean_err = sum(valid) / len(valid)
    mse = sum(e * e for e in valid) / len(valid)
    min_err = min(valid)
    max_err = max(valid)

    return {
        "mean": mean_err,
        "mse": mse,
        "min": min_err,
        "max": max_err,
    }


def _flatten_axes(axes, *, n_metrics: int, n_rows: int, n_cols: int):
    """Normalize subplot axes into a flat list for easy iteration."""
    if n_rows <= 1:
        # plt.subplots with a single row returns a 1D axes array.
        return list(axes)
    return [axes[r][c] for r in range(n_rows) for c in range(n_cols)]


def _draw_metric_unavailable(axis, metric: str) -> None:
    """Render a placeholder when a metric has no data on the given axis."""
    axis.text(
        0.5,
        0.5,
        f"{metric} not available",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    axis.set_ylabel(_metric_label(metric))
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.set_xticks([])


def _draw_single_metric(axis, module, records, metric: str, warnings: list[str]) -> None:
    """Draw one metric on the given axis, recording a warning if absent."""
    steps, values = module.extract_metric_series(records, metric)
    if not steps:
        warnings.append(f"metric '{metric}' not found in log")
        _draw_metric_unavailable(axis, metric)
        return
    axis.plot(steps, values, label=metric, linewidth=1.2)
    axis.set_ylabel(_metric_label(metric))
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(axis)


def _save_single_log_figure(plt, figure, *, output: Path) -> None:
    """Persist the single-log figure to disk."""
    try:
        figure.savefig(output, dpi=150, bbox_inches="tight")
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error


def _plot_single_log(
    plt,
    module,
    *,
    records,
    metrics: list[str],
    title: str,
    output: Path,
    no_show: bool,
):
    """Plot single log with 2-column layout."""
    selected = ["loss", "grad_norm"] + [metric for metric in metrics if metric not in {"loss", "grad_norm"}]
    n_metrics = len(selected)

    # Calculate grid: 2 columns, enough rows
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=False)
    axes_flat = _flatten_axes(axes, n_metrics=n_metrics, n_rows=n_rows, n_cols=n_cols)

    warnings: list[str] = []
    for idx, metric in enumerate(selected):
        _draw_single_metric(axes_flat[idx], module, records, metric, warnings)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    _save_single_log_figure(plt, figure, output=output)

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def _annotate_error_stats(axis, stats: dict[str, float]) -> None:
    """Annotate error statistics below an error-curve subplot."""
    stats_text = (
        f"mean={stats['mean']:.5f}, mse={stats['mse']:.5f}, "
        f"min={stats['min']:.5f}, max={stats['max']:.5f}"
    )
    axis.annotate(
        stats_text,
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color="navy",
    )


def _draw_main_metrics_row(axes, aligned, *, log_a_name: str, log_b_name: str) -> None:
    """Draw row 0: loss and grad_norm curves for both logs."""
    # Loss subplot
    ax_loss = axes[0][0]
    loss_a_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_a]
    loss_b_vals = [float(r.get("loss", float("nan"))) for r in aligned.records_b]
    ax_loss.plot(aligned.steps, loss_a_vals, label=log_a_name, linewidth=1.2)
    ax_loss.plot(aligned.steps, loss_b_vals, label=log_b_name, linewidth=1.2)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)
    ax_loss.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss)

    # Grad norm subplot
    ax_grad = axes[0][1]
    grad_a_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_a]
    grad_b_vals = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_b]
    ax_grad.plot(aligned.steps, grad_a_vals, label=log_a_name, linewidth=1.2)
    ax_grad.plot(aligned.steps, grad_b_vals, label=log_b_name, linewidth=1.2)
    ax_grad.set_ylabel("grad_norm")
    ax_grad.grid(True, linestyle="--", alpha=0.4)
    ax_grad.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad)


def _draw_abs_error_row(axes, aligned, *, loss_abs_err, loss_abs_stats, grad_abs_err, grad_abs_stats) -> None:
    """Draw row 1: absolute error curves (no threshold lines)."""
    # Loss abs error
    ax_loss_abs = axes[1][0]
    ax_loss_abs.plot(aligned.steps, loss_abs_err, label="loss abs error", color="blue", linewidth=1.0)
    ax_loss_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_abs.set_ylabel("loss abs error")
    ax_loss_abs.grid(True, linestyle="--", alpha=0.4)
    ax_loss_abs.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss_abs)
    _annotate_error_stats(ax_loss_abs, loss_abs_stats)

    # Grad norm abs error
    ax_grad_abs = axes[1][1]
    ax_grad_abs.plot(
        aligned.steps,
        grad_abs_err,
        label="grad_norm abs error",
        color="blue",
        linewidth=1.0,
    )
    ax_grad_abs.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_abs.set_ylabel("grad_norm abs error")
    ax_grad_abs.grid(True, linestyle="--", alpha=0.4)
    ax_grad_abs.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad_abs)
    _annotate_error_stats(ax_grad_abs, grad_abs_stats)


def _draw_rel_error_row(
    axes,
    aligned,
    *,
    baseline_name: str,
    loss_rel_err,
    loss_rel_stats,
    grad_rel_err,
    grad_rel_stats,
) -> None:
    """Draw row 2: relative error curves with threshold lines."""
    # Loss rel error with threshold lines
    ax_loss_rel = axes[2][0]
    ax_loss_rel.plot(
        aligned.steps,
        loss_rel_err,
        label=f"loss relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_loss_rel.axhline(y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02")
    ax_loss_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_loss_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_loss_rel.set_ylabel("loss relative error")
    ax_loss_rel.grid(True, linestyle="--", alpha=0.4)
    ax_loss_rel.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_loss_rel)
    _annotate_error_stats(ax_loss_rel, loss_rel_stats)

    # Grad norm rel error with threshold lines
    ax_grad_rel = axes[2][1]
    ax_grad_rel.plot(
        aligned.steps,
        grad_rel_err,
        label=f"grad_norm relative error (baseline={baseline_name})",
        color="green",
        linewidth=1.0,
    )
    ax_grad_rel.axhline(y=0.02, color="red", linestyle="--", linewidth=1.5, label="threshold ±0.02")
    ax_grad_rel.axhline(y=-0.02, color="red", linestyle="--", linewidth=1.5)
    ax_grad_rel.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax_grad_rel.set_ylabel("grad_norm relative error")
    ax_grad_rel.grid(True, linestyle="--", alpha=0.4)
    ax_grad_rel.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(ax_grad_rel)
    _annotate_error_stats(ax_grad_rel, grad_rel_stats)


def _draw_optional_compare_metric(
    axis,
    aligned,
    metric: str,
    *,
    log_a_name: str,
    log_b_name: str,
    warnings: list[str],
) -> None:
    """Draw one optional metric comparison on the given axis."""
    values_a = []
    values_b = []
    for record_a, record_b in zip(aligned.records_a, aligned.records_b, strict=True):
        value_a = record_a.get(metric)
        value_b = record_b.get(metric)
        values_a.append(float(value_a) if value_a is not None else float("nan"))
        values_b.append(float(value_b) if value_b is not None else float("nan"))

    has_valid_a = any(not math.isnan(value) for value in values_a)
    has_valid_b = any(not math.isnan(value) for value in values_b)
    if not has_valid_a and not has_valid_b:
        warnings.append(f"metric '{metric}' not found in either log")
        _draw_metric_unavailable(axis, metric)
        return

    axis.plot(aligned.steps, values_a, label=log_a_name, linewidth=1.2)
    axis.plot(aligned.steps, values_b, label=log_b_name, linewidth=1.2)
    axis.set_ylabel(_metric_label(metric))
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="best", fontsize=8)
    _apply_integer_xaxis(axis)


def _draw_optional_compare_metrics(
    axes,
    aligned,
    optional_metrics: list[str],
    *,
    n_rows: int,
    n_optional_rows: int,
    log_a_name: str,
    log_b_name: str,
    warnings: list[str],
) -> None:
    """Draw the optional-metric comparison rows and hide unused subplots."""
    for idx, metric in enumerate(optional_metrics):
        row = 3 + idx // 2
        col = idx % 2
        _draw_optional_compare_metric(
            axes[row][col],
            aligned,
            metric,
            log_a_name=log_a_name,
            log_b_name=log_b_name,
            warnings=warnings,
        )

    # Hide unused subplots in optional metrics section
    n_optional_plotted = len(optional_metrics)
    for idx in range(n_optional_plotted, n_optional_rows * 2):
        row = 3 + idx // 2
        col = idx % 2
        if row < n_rows:
            axes[row][col].axis("off")


def _save_compare_figure(plt, figure, *, output: Path) -> None:
    """Persist the comparison figure to disk."""
    try:
        figure.savefig(output, dpi=150, bbox_inches="tight")
    except OSError as error:
        plt.close(figure)
        raise RuntimeError(f"failed to save plot to '{output}': {error}") from error


def _compute_compare_error_stats(module, aligned, baseline: str):
    """Compute signed errors and statistics for loss and grad_norm."""
    # Compute signed errors for loss and grad_norm
    losses_a = [float(r["loss"]) for r in aligned.records_a]
    losses_b = [float(r["loss"]) for r in aligned.records_b]
    loss_abs_err, loss_rel_err = module.compute_signed_errors(losses_a, losses_b, baseline=baseline)

    grad_norms_a = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_a]
    grad_norms_b = [float(r.get("grad_norm", float("nan"))) for r in aligned.records_b]
    grad_abs_err, grad_rel_err = module.compute_signed_errors(grad_norms_a, grad_norms_b, baseline=baseline)

    # Compute error statistics
    loss_abs_stats = _compute_error_stats(loss_abs_err)
    loss_rel_stats = _compute_error_stats(loss_rel_err)
    grad_abs_stats = _compute_error_stats(grad_abs_err)
    grad_rel_stats = _compute_error_stats(grad_rel_err)
    return {
        "loss_abs_err": loss_abs_err,
        "loss_rel_err": loss_rel_err,
        "grad_abs_err": grad_abs_err,
        "grad_rel_err": grad_rel_err,
        "loss_abs_stats": loss_abs_stats,
        "loss_rel_stats": loss_rel_stats,
        "grad_abs_stats": grad_abs_stats,
        "grad_rel_stats": grad_rel_stats,
    }


def _draw_compare_error_rows(axes, aligned, *, error_stats, baseline_name: str) -> None:
    """Draw absolute and relative error rows for loss and grad_norm."""
    _draw_abs_error_row(
        axes,
        aligned,
        loss_abs_err=error_stats["loss_abs_err"],
        loss_abs_stats=error_stats["loss_abs_stats"],
        grad_abs_err=error_stats["grad_abs_err"],
        grad_abs_stats=error_stats["grad_abs_stats"],
    )
    _draw_rel_error_row(
        axes,
        aligned,
        baseline_name=baseline_name,
        loss_rel_err=error_stats["loss_rel_err"],
        loss_rel_stats=error_stats["loss_rel_stats"],
        grad_rel_err=error_stats["grad_rel_err"],
        grad_rel_stats=error_stats["grad_rel_stats"],
    )


def _plot_compare(
    plt,
    module,
    *,
    records_a,
    records_b,
    metrics: list[str],
    baseline: str,
    title: str,
    output: Path,
    no_show: bool,
    log_a_name: str = "a",
    log_b_name: str = "b",
):
    """Plot comparison with 2-column layout and error curves under each metric."""
    aligned = module.align_by_common_steps(records_a, records_b)
    warnings: list[str] = []
    if aligned.missing_in_a:
        warnings.append(f"steps only in log-b: {len(aligned.missing_in_a)}")
    if aligned.missing_in_b:
        warnings.append(f"steps only in log-a: {len(aligned.missing_in_b)}")
    if not aligned.steps:
        warnings.append("no common steps between log-a and log-b")
        return warnings

    # Required metrics with error tracking
    required_metrics = ["loss", "grad_norm"]
    optional_metrics = [m for m in metrics if m not in required_metrics]

    error_stats = _compute_compare_error_stats(module, aligned, baseline)

    # Build subplot list: each main metric followed by its error curves
    # Layout: 2 columns
    # Row 0: loss, grad_norm
    # Row 1: loss_abs_error, grad_norm_abs_error
    # Row 2: loss_rel_error, grad_norm_rel_error
    # Then optional metrics in subsequent rows

    n_optional_rows = (len(optional_metrics) + 1) // 2
    n_rows = 3 + n_optional_rows  # 3 rows for loss+grad_norm with their errors

    figure, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))

    _draw_main_metrics_row(axes, aligned, log_a_name=log_a_name, log_b_name=log_b_name)
    baseline_name = log_a_name if baseline == "a" else log_b_name
    _draw_compare_error_rows(
        axes,
        aligned,
        error_stats=error_stats,
        baseline_name=baseline_name,
    )
    _draw_optional_compare_metrics(
        axes,
        aligned,
        optional_metrics,
        n_rows=n_rows,
        n_optional_rows=n_optional_rows,
        log_a_name=log_a_name,
        log_b_name=log_b_name,
        warnings=warnings,
    )

    # Set xlabel for bottom row
    for col in range(2):
        axes[-1][col].set_xlabel("step")

    figure.suptitle(title, fontsize=12)
    figure.tight_layout(rect=(0.02, 0.03, 1, 0.97))
    _save_compare_figure(plt, figure, output=output)

    if not no_show:
        plt.show()
    plt.close(figure)
    return warnings


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training metrics from torchtitan-style training logs")
    parser.add_argument("--log-a", required=True, help="path to primary log")
    parser.add_argument("--log-b", default=None, help="path to comparison log")
    parser.add_argument(
        "--metrics",
        default="",
        help=(
            "optional metrics: memory(memory_gib),memory_pct,tps,tflops,"
            "mfu(mfu_pct),elapsed(elapsed_time_per_step),indexer(indexer_loss)"
        ),
    )
    parser.add_argument("--output", default=None, help="output image path")
    parser.add_argument("--title", default=None, help="figure title")
    parser.add_argument(
        "--baseline",
        choices=["a", "b"],
        default="a",
        help="baseline for relative error (default: a, i.e., log-a)",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png", help="output format")
    parser.add_argument("--no-show", action="store_true", help="disable interactive display")
    return parser


def _read_log(module, log_path: str, label: str):
    """Read a single log file, returning (records, warnings) or None on failure."""
    try:
        records, log_warnings = module.read_training_metrics(log_path)
    except OSError as error:
        logger.info(f"[error] failed to read {label} '{log_path}': {error}")
        return None
    if not records:
        logger.info(f"[error] no valid training metrics found in {label}: {log_path}")
        return None
    return records, log_warnings


def _resolve_output_path(args) -> Path:
    output = Path(args.output) if args.output else _default_output_path(args.log_a, args.log_b, args.format)
    if output.suffix.lower() != f".{args.format}":
        output = output.with_suffix(f".{args.format}")
    return output


def _dispatch_plot(
    plt,
    module,
    *,
    args,
    records_a,
    records_b,
    metrics,
    title,
    output,
    log_a_name,
    log_b_name,
):
    """Dispatch to single-log or comparison plotting and return warnings."""
    if records_b is None:
        return _plot_single_log(
            plt,
            module,
            records=records_a,
            metrics=metrics,
            title=title,
            output=output,
            no_show=args.no_show,
        )
    return _plot_compare(
        plt,
        module,
        records_a=records_a,
        records_b=records_b,
        metrics=metrics,
        baseline=args.baseline,
        title=title,
        output=output,
        no_show=args.no_show,
        log_a_name=log_a_name,
        log_b_name=log_b_name or "b",
    )


def _log_run_summary(module, *, output, records_a, records_b, warnings) -> None:
    """Emit the success/summary/warning log lines after a successful plot."""
    logger.info(f"[ok] plot saved to: {output}")
    summary_a = module.summarize_records(records_a)
    logger.info(
        f"[summary-a] steps={summary_a['num_steps']} "
        f"range=[{summary_a['first_step']},{summary_a['last_step']}] "
        f"loss={summary_a['first_loss']:.5f}->{summary_a['last_loss']:.5f} "
        f"grad_norm={summary_a['first_grad_norm']:.4f}->{summary_a['last_grad_norm']:.4f}"
    )
    if records_b is not None:
        summary_b = module.summarize_records(records_b)
        logger.info(
            f"[summary-b] steps={summary_b['num_steps']} "
            f"range=[{summary_b['first_step']},{summary_b['last_step']}] "
            f"loss={summary_b['first_loss']:.5f}->{summary_b['last_loss']:.5f} "
            f"grad_norm={summary_b['first_grad_norm']:.4f}->{summary_b['last_grad_norm']:.4f}"
        )

    for warning in warnings:
        logger.info(f"[warning] {warning}")


def _read_both_logs(module, args):
    """Read log-a and optional log-b, returning (records_a, records_b, warnings) or None."""
    read_a = _read_log(module, args.log_a, "log-a")
    if read_a is None:
        return None
    records_a, warnings_a = read_a

    records_b = None
    warnings_b: list[str] = []
    if args.log_b:
        read_b = _read_log(module, args.log_b, "log-b")
        if read_b is None:
            return None
        records_b, warnings_b = read_b

    return records_a, records_b, [*warnings_a, *warnings_b]


class _PlotContext(NamedTuple):
    output: Path
    metrics: list[str]
    log_a_name: str
    log_b_name: str | None
    title: str


def _resolve_plot_context(args) -> "_PlotContext":
    """Resolve output paths, metrics, log names and title from parsed args."""
    output = _resolve_output_path(args)

    metrics = _parse_metric_list(args.metrics)

    # Get log file names for legend
    log_a_name = Path(args.log_a).stem
    log_b_name = Path(args.log_b).stem if args.log_b else None

    title = args.title or (
        f"Training Metrics: {log_a_name} vs {log_b_name}" if args.log_b else f"Training Metrics: {log_a_name}"
    )
    return _PlotContext(output, metrics, log_a_name, log_b_name, title)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    module = _load_train_log_metrics_module()

    read_logs = _read_both_logs(module, args)
    if read_logs is None:
        return 1
    records_a, records_b, warnings = read_logs

    output, metrics, log_a_name, log_b_name, title = _resolve_plot_context(args)

    try:
        plt = _require_matplotlib(args.no_show)
    except RuntimeError as error:
        logger.info(f"[error] {error}")
        return 1

    try:
        warnings.extend(
            _dispatch_plot(
                plt,
                module,
                args=args,
                records_a=records_a,
                records_b=records_b,
                metrics=metrics,
                title=title,
                output=output,
                log_a_name=log_a_name,
                log_b_name=log_b_name,
            )
        )
    except RuntimeError as error:
        logger.info(f"[error] {error}")
        return 1

    if records_b is not None and "no common steps between log-a and log-b" in warnings:
        logger.info("[error] no common steps between log-a and log-b")
        return 1

    _log_run_summary(
        module,
        output=output,
        records_a=records_a,
        records_b=records_b,
        warnings=warnings,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
