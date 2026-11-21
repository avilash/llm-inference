#!/usr/bin/env python3
"""
plot_results.py - Visualizes and aggregates LLM benchmark serving results & GPU memory.

Features:
  - Compiles benchmark results across concurrency sweeps.
  - Outputs formatted comparative Markdown/text tables including Peak VRAM & VRAM/Stream.
  - Generates ASCII terminal charts for fast CLI analysis (Throughput scaling, TTFT vs TPS, Peak VRAM).
  - Exports PNG graphs if matplotlib is installed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_results_from_files(files: List[Path]) -> List[Dict[str, Any]]:
    """Load benchmark summaries from single benchmark files or sweep summary files."""
    summaries = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if "summary" in item:
                                summaries.append(item["summary"])
                            elif "concurrency" in item:
                                summaries.append(item)
                elif isinstance(content, dict):
                    if "summary" in content:
                        summaries.append(content["summary"])
                    elif "concurrency" in content:
                        summaries.append(content)
        except Exception as e:
            print(f"[Warning] Could not load {fpath}: {e}", file=sys.stderr)

    # Deduplicate and sort by concurrency
    unique_summaries = {}
    for s in summaries:
        c = s.get("concurrency", 0)
        unique_summaries[c] = s

    sorted_summaries = [unique_summaries[c] for c in sorted(unique_summaries.keys())]
    return sorted_summaries


def print_markdown_table(summaries: List[Dict[str, Any]]):
    """Print comparative markdown table across concurrency levels including GPU memory."""
    has_gpu = any(s.get("gpu_available") for s in summaries)

    if has_gpu:
        headers = [
            "Concurrency",
            "Req/s",
            "Tok/s (Out)",
            "TTFT p50 (s)",
            "TTFT p99 (s)",
            "ITL p50 (ms)",
            "Peak VRAM (MB)",
            "VRAM / Stream",
            "E2E p50 (s)",
        ]
        row_fmt = "| {:^11} | {:^7} | {:^11} | {:^12} | {:^12} | {:^12} | {:^14} | {:^13} | {:^11} |"
        divider = "|-" + "-|-".join(["-" * 11, "-" * 7, "-" * 11, "-" * 12, "-" * 12, "-" * 12, "-" * 14, "-" * 13, "-" * 11]) + "-|"
    else:
        headers = [
            "Concurrency",
            "Req/s",
            "Tok/s (Out)",
            "Tokens/s (Tot)",
            "TTFT p50 (s)",
            "TTFT p99 (s)",
            "ITL p50 (ms)",
            "ITL p99 (ms)",
            "E2E p50 (s)",
        ]
        row_fmt = "| {:^11} | {:^7} | {:^11} | {:^14} | {:^12} | {:^12} | {:^12} | {:^12} | {:^11} |"
        divider = "|-" + "-|-".join(["-" * 11, "-" * 7, "-" * 11, "-" * 14, "-" * 12, "-" * 12, "-" * 12, "-" * 12, "-" * 11]) + "-|"

    total_width = len(divider)
    print("\n" + "=" * total_width)
    print("                    CONCURRENCY SCALING COMPARATIVE ANALYSIS TABLE".center(total_width))
    print("=" * total_width)
    print(row_fmt.format(*headers))
    print(divider)

    for s in summaries:
        if has_gpu:
            peak_vram_str = f"{s.get('gpu_peak_mb', 0.0):,.1f}" if s.get('gpu_peak_mb') is not None else "-"
            per_stream_str = f"{s.get('gpu_per_stream_mb', 0.0):.1f} MB" if s.get('gpu_per_stream_mb') is not None else "-"
            row = [
                str(s.get("concurrency", "-")),
                f"{s.get('request_throughput_rps', 0.0):.2f}",
                f"{s.get('output_token_throughput_tps', 0.0):.2f}",
                f"{s.get('ttft_median_s', 0.0):.4f}",
                f"{s.get('ttft_p99_s', 0.0):.4f}",
                f"{s.get('itl_median_ms', 0.0):.2f}",
                peak_vram_str,
                per_stream_str,
                f"{s.get('e2e_median_s', 0.0):.4f}",
            ]
        else:
            row = [
                str(s.get("concurrency", "-")),
                f"{s.get('request_throughput_rps', 0.0):.2f}",
                f"{s.get('output_token_throughput_tps', 0.0):.2f}",
                f"{s.get('total_token_throughput_tps', 0.0):.2f}",
                f"{s.get('ttft_median_s', 0.0):.4f}",
                f"{s.get('ttft_p99_s', 0.0):.4f}",
                f"{s.get('itl_median_ms', 0.0):.2f}",
                f"{s.get('itl_p99_ms', 0.0):.2f}",
                f"{s.get('e2e_median_s', 0.0):.4f}",
            ]
        print(row_fmt.format(*row))
    print("=" * total_width + "\n")


def draw_ascii_chart(title: str, x_labels: List[str], y_vals: List[float], y_unit: str = ""):
    """Render a clean terminal ASCII chart."""
    if not y_vals or max(y_vals) == 0:
        return

    max_val = max(y_vals)
    min_val = min(y_vals)
    height = 10
    width = len(y_vals)

    print(f"\n--- [Chart] {title} ({y_unit}) ---")
    val_range = max_val - min_val if max_val != min_val else 1.0

    for row in range(height, -1, -1):
        level = min_val + (row / height) * val_range
        label = f"{level:7.1f} |"
        line = ""
        for val in y_vals:
            bar_height = int(round(((val - min_val) / val_range) * height))
            if bar_height == row:
                line += "  *   "
            elif bar_height > row:
                line += "  |   "
            else:
                line += "      "
        print(f"{label}{line}")

    print(" " * 8 + "+-" + "------" * width)
    x_line = " " * 9
    for xl in x_labels:
        x_line += f"{xl:^6}"
    print(x_line)
    print(" " * 9 + "Concurrency (N)\n")


def try_matplotlib_plots(summaries: List[Dict[str, Any]], out_dir: Path):
    """Generate high-resolution PNG charts if matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Notice] Matplotlib is not installed. Skipping PNG image generation (ASCII charts rendered above).")
        return

    concurrencies = [s["concurrency"] for s in summaries]
    out_tps = [s["output_token_throughput_tps"] for s in summaries]
    ttft_p50 = [s["ttft_median_s"] for s in summaries]
    ttft_p99 = [s["ttft_p99_s"] for s in summaries]
    itl_p50 = [s["itl_median_ms"] for s in summaries]

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Throughput Scaling Curve
    plt.figure(figsize=(8, 5))
    plt.plot(concurrencies, out_tps, marker="o", color="#2563EB", linewidth=2.5, markersize=8)
    plt.title("Output Token Throughput Scaling vs. Concurrency", fontsize=14, fontweight="bold")
    plt.xlabel("Concurrency (Number of Parallel Clients)", fontsize=11)
    plt.ylabel("Token Throughput (Tokens / sec)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(concurrencies)
    tps_file = out_dir / "throughput_scaling.png"
    plt.savefig(tps_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {tps_file}")

    # 2. Pareto Curve: TTFT vs Output Token Throughput
    plt.figure(figsize=(8, 5))
    plt.plot(out_tps, ttft_p50, marker="s", color="#DC2626", linewidth=2.5, markersize=8, label="TTFT Median (p50)")
    plt.plot(out_tps, ttft_p99, marker="^", color="#F59E0B", linewidth=2.0, linestyle="--", markersize=7, label="TTFT Tail (p99)")
    for i, c in enumerate(concurrencies):
        plt.annotate(f"N={c}", (out_tps[i], ttft_p50[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    plt.title("Pareto Curve: TTFT vs. Token Throughput", fontsize=14, fontweight="bold")
    plt.xlabel("Throughput (Tokens / sec)", fontsize=11)
    plt.ylabel("Time To First Token (seconds)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    pareto_file = out_dir / "pareto_ttft_vs_throughput.png"
    plt.savefig(pareto_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {pareto_file}")

    # 3. Peak VRAM Scaling Curve (if GPU metrics exist)
    has_vram = any(s.get("gpu_peak_mb") is not None for s in summaries)
    if has_vram:
        vram_vals = [s.get("gpu_peak_mb", 0.0) for s in summaries]
        plt.figure(figsize=(8, 5))
        plt.plot(concurrencies, vram_vals, marker="D", color="#7C3AED", linewidth=2.5, markersize=8)
        plt.title("Peak GPU VRAM Usage vs. Concurrency", fontsize=14, fontweight="bold")
        plt.xlabel("Concurrency (Number of Parallel Clients)", fontsize=11)
        plt.ylabel("Peak VRAM Allocated (MB)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(concurrencies)
        vram_file = out_dir / "vram_scaling.png"
        plt.savefig(vram_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot Saved] {vram_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot & Aggregate LLM Benchmark Results")
    parser.add_argument("--results-dir", type=str, default="benchmarks/results", help="Directory containing benchmark JSON result files")
    parser.add_argument("--files", nargs="*", default=None, help="Specific JSON result files to plot")
    parser.add_argument("--output-dir", type=str, default="benchmarks/plots", help="Directory to save generated plot PNGs")

    args = parser.parse_args()

    files_to_load = []
    if args.files:
        files_to_load = [Path(f) for f in args.files]
    else:
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            files_to_load = list(results_dir.glob("benchmark_*.json"))
            if not files_to_load:
                files_to_load = list(results_dir.glob("sweep_summary_*.json"))

    if not files_to_load:
        print(f"[Error] No benchmark result JSON files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = load_results_from_files(files_to_load)
    if not summaries:
        print("[Error] No valid benchmark summaries could be loaded.", file=sys.stderr)
        sys.exit(1)

    # 1. Print comparative markdown table
    print_markdown_table(summaries)

    # 2. Render ASCII terminal charts
    concurrencies = [str(s.get("concurrency", 0)) for s in summaries]
    tps_vals = [s.get("output_token_throughput_tps", 0.0) for s in summaries]
    ttft_vals = [s.get("ttft_median_s", 0.0) for s in summaries]
    vram_vals = [s.get("gpu_peak_mb", 0.0) for s in summaries if s.get("gpu_peak_mb") is not None]

    draw_ascii_chart("Output Token Throughput Scaling", concurrencies, tps_vals, y_unit="tokens/s")
    draw_ascii_chart("Time To First Token (TTFT Median)", concurrencies, ttft_vals, y_unit="seconds")
    if vram_vals and len(vram_vals) == len(concurrencies):
        draw_ascii_chart("Peak GPU VRAM Allocated", concurrencies, vram_vals, y_unit="MB")

    # 3. Export PNG plots if matplotlib is available
    out_dir = Path(args.output_dir)
    try_matplotlib_plots(summaries, out_dir)


if __name__ == "__main__":
    main()
