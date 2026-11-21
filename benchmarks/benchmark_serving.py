#!/usr/bin/env python3
"""
benchmark_serving.py - Client-side LLM Inference Benchmarking & Memory Profiling Suite.

Measures:
  - Time To First Token (TTFT)
  - Inter-Token Latency (ITL) / Time Per Output Token (TPOT)
  - End-to-End Latency (E2E)
  - Request Throughput (req/s)
  - Token Throughput (output tokens/s & total tokens/s)
  - Percentiles (p50, p90, p95, p99) under configurable concurrency or Poisson arrival.
  - Client-Side GPU Memory Profiling (Baseline VRAM, Peak VRAM, Delta VRAM, VRAM/Stream, Residual Leaks).

Zero server-side modifications required (runs purely out-of-process via NVML / nvidia-smi).
"""

import argparse
import asyncio
import ctypes
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np


class GPUMemoryMonitor:
    """
    Asynchronous, non-blocking GPU memory sampler.
    Queries GPU VRAM directly via NVML (ctypes to libnvidia-ml.so) or nvidia-smi fallback.
    """

    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.05):
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s
        self.is_available = False
        self.device_name: Optional[str] = None
        self.total_memory_mb: Optional[float] = None

        self._nvml = None
        self._handle = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        self.samples: List[Tuple[float, float]] = []  # (timestamp, used_mb)
        self.baseline_used_mb: Optional[float] = None
        self.peak_used_mb: Optional[float] = None
        self.final_used_mb: Optional[float] = None

        self._init_backend()

    def _init_backend(self):
        """Initialize NVML via ctypes for high-performance sub-millisecond memory queries."""
        try:
            for lib_name in ["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"]:
                try:
                    self._nvml = ctypes.CDLL(lib_name)
                    break
                except OSError:
                    continue

            if self._nvml:
                ret = self._nvml.nvmlInit()
                if ret == 0:
                    handle = ctypes.c_void_p()
                    ret_h = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index, ctypes.byref(handle))
                    if ret_h == 0:
                        self._handle = handle
                        name_buf = ctypes.create_string_buffer(64)
                        self._nvml.nvmlDeviceGetName(self._handle, name_buf, 64)
                        self.device_name = name_buf.value.decode("utf-8", errors="ignore")

                        # Get total memory
                        mem_info = self._get_nvml_memory()
                        if mem_info:
                            self.total_memory_mb = round(mem_info[1], 2)
                            self.is_available = True
                            return
        except Exception:
            pass

        # Fallback: check nvidia-smi command
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=2.0,
                ).strip()
                parts = [p.strip() for p in out.split(",")]
                self.device_name = parts[0]
                self.total_memory_mb = float(parts[1])
                self.is_available = True
            except Exception:
                self.is_available = False

    def _get_nvml_memory(self) -> Optional[Tuple[float, float]]:
        """Return (used_mb, total_mb) via NVML."""
        if not self._nvml or not self._handle:
            return None

        class _nvmlMemory_t(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        mem = _nvmlMemory_t()
        ret = self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(mem))
        if ret == 0:
            return (mem.used / (1024 * 1024), mem.total / (1024 * 1024))
        return None

    def query_memory_mb(self) -> Optional[float]:
        """Fetch current used VRAM in MB."""
        if not self.is_available:
            return None

        # 1. Try NVML fast path
        mem_info = self._get_nvml_memory()
        if mem_info:
            return round(mem_info[0], 2)

        # 2. Subprocess fallback
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=2.0,
            ).strip()
            return float(out)
        except Exception:
            return None

    async def _sample_loop(self):
        """Periodic background sampler."""
        while self._running:
            used = self.query_memory_mb()
            if used is not None:
                self.samples.append((time.perf_counter(), used))
            await asyncio.sleep(self.sample_interval_s)

    async def __aenter__(self):
        self.samples.clear()
        self.baseline_used_mb = self.query_memory_mb()
        if self.is_available:
            self._running = True
            self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.final_used_mb = self.query_memory_mb()
        if self.samples:
            all_used = [s[1] for s in self.samples]
            self.peak_used_mb = max(all_used)
        else:
            self.peak_used_mb = self.baseline_used_mb

    def get_metrics(self, concurrency: int) -> Dict[str, Any]:
        """Compute aggregated GPU memory stats."""
        if not self.is_available or self.baseline_used_mb is None or self.peak_used_mb is None:
            return {
                "gpu_available": False,
                "gpu_device_name": None,
                "gpu_total_mb": None,
                "gpu_baseline_mb": None,
                "gpu_peak_mb": None,
                "gpu_delta_mb": None,
                "gpu_per_stream_mb": None,
                "gpu_residual_mb": None,
            }

        delta_mb = max(0.0, self.peak_used_mb - self.baseline_used_mb)
        per_stream_mb = delta_mb / concurrency if concurrency > 0 else 0.0
        residual_mb = (self.final_used_mb - self.baseline_used_mb) if self.final_used_mb is not None else 0.0

        return {
            "gpu_available": True,
            "gpu_device_name": self.device_name,
            "gpu_total_mb": self.total_memory_mb,
            "gpu_baseline_mb": round(self.baseline_used_mb, 2),
            "gpu_peak_mb": round(self.peak_used_mb, 2),
            "gpu_delta_mb": round(delta_mb, 2),
            "gpu_per_stream_mb": round(per_stream_mb, 2),
            "gpu_residual_mb": round(residual_mb, 2),
        }


@dataclass
class RequestMetric:
    request_id: str
    prompt: str
    prompt_len: int
    output_len: int
    ttft: float
    itl_list: List[float] = field(default_factory=list)
    e2e_latency: float = 0.0
    status_code: int = 200
    success: bool = True
    error_msg: Optional[str] = None


@dataclass
class BenchmarkSummary:
    timestamp: str
    model_name: str
    endpoint: str
    concurrency: int
    request_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    total_token_throughput_tps: float
    # TTFT metrics (seconds)
    ttft_mean_s: float
    ttft_median_s: float
    ttft_p90_s: float
    ttft_p95_s: float
    ttft_p99_s: float
    ttft_min_s: float
    ttft_max_s: float
    # ITL metrics (milliseconds)
    itl_mean_ms: float
    itl_median_ms: float
    itl_p90_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    itl_min_ms: float
    itl_max_ms: float
    # E2E Latency metrics (seconds)
    e2e_mean_s: float
    e2e_median_s: float
    e2e_p90_s: float
    e2e_p95_s: float
    e2e_p99_s: float
    e2e_min_s: float
    e2e_max_s: float
    # GPU Memory Profiling metrics (MB)
    gpu_available: bool = False
    gpu_device_name: Optional[str] = None
    gpu_total_mb: Optional[float] = None
    gpu_baseline_mb: Optional[float] = None
    gpu_peak_mb: Optional[float] = None
    gpu_delta_mb: Optional[float] = None
    gpu_per_stream_mb: Optional[float] = None
    gpu_residual_mb: Optional[float] = None


def load_dataset_prompts(dataset_path: Path, num_prompts: int) -> List[Dict[str, Any]]:
    """Load prompts from a JSON dataset file (e.g. sample_prompts.json or ShareGPT)."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            if "prompt" in item:
                prompts.append({
                    "id": item.get("id", f"prompt-{idx}"),
                    "prompt": item["prompt"],
                    "max_tokens": item.get("max_tokens", 128),
                })
            elif "conversations" in item:
                convs = item["conversations"]
                for msg in convs:
                    if msg.get("from") in ("human", "user"):
                        prompts.append({
                            "id": item.get("id", f"sharegpt-{idx}"),
                            "prompt": msg["value"],
                            "max_tokens": 128,
                        })
                        break

    if not prompts:
        raise ValueError(f"No valid prompts found in dataset {dataset_path}")

    sampled = []
    for i in range(num_prompts):
        sampled.append(prompts[i % len(prompts)])
    return sampled


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    req_data: Dict[str, Any],
    is_chat: bool,
    stream: bool = True,
    model_name: Optional[str] = None,
    start_time: Optional[float] = None,
) -> RequestMetric:
    """Send an inference request (streaming SSE or non-streaming) and record high-precision latencies."""
    prompt_text = req_data["prompt"]
    max_tokens = req_data.get("max_tokens", 128)
    req_id = req_data.get("id", f"req-{random.randint(1000, 9999)}")

    payload = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        **({"messages": [{"role": "user", "content": prompt_text}]} if is_chat else {"prompt": prompt_text}),
    }
    if model_name:
        payload["model"] = model_name

    prompt_estimated_tokens = len(prompt_text.split())
    t_start = start_time if start_time is not None else time.perf_counter()

    try:
        if stream:
            output_tokens_count = 0
            token_timestamps: List[float] = []
            t_first_token: Optional[float] = None

            async with client.stream("POST", url, headers=headers, json=payload, timeout=120.0) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    return RequestMetric(
                        request_id=req_id,
                        prompt=prompt_text,
                        prompt_len=prompt_estimated_tokens,
                        output_len=0,
                        ttft=0.0,
                        status_code=response.status_code,
                        success=False,
                        error_msg=f"HTTP {response.status_code}: {err_body.decode('utf-8', errors='ignore')}",
                    )

                async for line in response.aiter_lines():
                    t_now = time.perf_counter()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta_content = ""
                            if "delta" in choice:
                                delta_content = choice["delta"].get("content", "")
                            elif "text" in choice:
                                delta_content = choice.get("text", "")

                            if delta_content:
                                if t_first_token is None:
                                    t_first_token = t_now
                                token_timestamps.append(t_now)
                                output_tokens_count += 1
                    except json.JSONDecodeError:
                        continue

            t_end = time.perf_counter()
            e2e_latency = t_end - t_start
            ttft = (t_first_token - t_start) if t_first_token is not None else e2e_latency

            itl_list = []
            if len(token_timestamps) > 1:
                for i in range(1, len(token_timestamps)):
                    itl_list.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000.0)

            return RequestMetric(
                request_id=req_id,
                prompt=prompt_text,
                prompt_len=prompt_estimated_tokens,
                output_len=output_tokens_count,
                ttft=ttft,
                itl_list=itl_list,
                e2e_latency=e2e_latency,
                status_code=200,
                success=True,
            )
        else:
            response = await client.post(url, headers=headers, json=payload, timeout=120.0)
            t_end = time.perf_counter()
            e2e_latency = t_end - t_start

            if response.status_code != 200:
                return RequestMetric(
                    request_id=req_id,
                    prompt=prompt_text,
                    prompt_len=prompt_estimated_tokens,
                    output_len=0,
                    ttft=e2e_latency,
                    status_code=response.status_code,
                    success=False,
                    error_msg=response.text,
                )

            resp_json = response.json()
            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", prompt_estimated_tokens)
            output_tokens = usage.get("completion_tokens", 0)

            return RequestMetric(
                request_id=req_id,
                prompt=prompt_text,
                prompt_len=prompt_tokens,
                output_len=output_tokens,
                ttft=e2e_latency,
                itl_list=[],
                e2e_latency=e2e_latency,
                status_code=200,
                success=True,
            )

    except Exception as exc:
        return RequestMetric(
            request_id=req_id,
            prompt=prompt_text,
            prompt_len=prompt_estimated_tokens,
            output_len=0,
            ttft=0.0,
            status_code=0,
            success=False,
            error_msg=str(exc),
        )


async def run_benchmark(
    base_url: str,
    endpoint: str,
    api_key: str,
    prompts: List[Dict[str, Any]],
    concurrency: int,
    request_rate: float,
    stream: bool,
    model_name: Optional[str] = None,
) -> Tuple[BenchmarkSummary, List[RequestMetric]]:
    """Execute concurrent load test with out-of-process client-side GPU memory sampling."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    is_chat = "chat" in endpoint

    limits = httpx.Limits(max_connections=max(concurrency * 2, 100), max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        try:
            health_resp = await client.get(f"{base_url.rstrip('/')}/health")
            if health_resp.status_code != 200:
                print(f"[Warning] Health check returned HTTP {health_resp.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"[Error] Could not connect to {base_url}: {e}", file=sys.stderr)
            raise

        results: List[RequestMetric] = []
        gpu_monitor = GPUMemoryMonitor()

        # Execute load test wrapped inside asynchronous GPU memory sampler
        async with gpu_monitor:
            t_bench_start = time.perf_counter()

            if math.isinf(request_rate) or request_rate <= 0:
                # Fixed Concurrency Worker Pool
                queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
                for p in prompts:
                    queue.put_nowait(p)

                async def worker():
                    while not queue.empty():
                        try:
                            p = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        res = await send_request(
                            client, url, headers, p, is_chat, stream=stream, model_name=model_name
                        )
                        results.append(res)
                        queue.task_done()

                workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
                await asyncio.gather(*workers)
            else:
                # Poisson Arrival Distribution
                semaphore = asyncio.Semaphore(concurrency)

                async def sem_task(p: Dict[str, Any], t_sched: float):
                    async with semaphore:
                        res = await send_request(
                            client, url, headers, p, is_chat,
                            stream=stream, model_name=model_name, start_time=t_sched
                        )
                        results.append(res)

                tasks = []
                for p in prompts:
                    tasks.append(asyncio.create_task(sem_task(p, time.perf_counter())))
                    await asyncio.sleep(-math.log(1.0 - random.random()) / request_rate)
                await asyncio.gather(*tasks)

            t_bench_end = time.perf_counter()
            total_duration = t_bench_end - t_bench_start

    # Fetch GPU memory profiling metrics
    gpu_stats = gpu_monitor.get_metrics(concurrency)

    # Compute Statistics
    successful_requests = [r for r in results if r.success]
    failed_requests = [r for r in results if not r.success]

    total_prompts = len(results)
    success_count = len(successful_requests)
    fail_count = len(failed_requests)

    total_output_tokens = sum(r.output_len for r in successful_requests)
    total_prompt_tokens = sum(r.prompt_len for r in successful_requests)

    req_tps = success_count / total_duration if total_duration > 0 else 0.0
    out_tok_tps = total_output_tokens / total_duration if total_duration > 0 else 0.0
    tot_tok_tps = (total_prompt_tokens + total_output_tokens) / total_duration if total_duration > 0 else 0.0

    ttfts = [r.ttft for r in successful_requests] if successful_requests else [0.0]
    e2es = [r.e2e_latency for r in successful_requests] if successful_requests else [0.0]
    all_itls = []
    for r in successful_requests:
        all_itls.extend(r.itl_list)
    if not all_itls:
        all_itls = [0.0]

    def stats(arr: List[float]):
        np_arr = np.array(arr)
        return (
            float(np.mean(np_arr)),
            float(np.percentile(np_arr, 50)),
            float(np.percentile(np_arr, 90)),
            float(np.percentile(np_arr, 95)),
            float(np.percentile(np_arr, 99)),
            float(np.min(np_arr)),
            float(np.max(np_arr)),
        )

    ttft_mean, ttft_p50, ttft_p90, ttft_p95, ttft_p99, ttft_min, ttft_max = stats(ttfts)
    itl_mean, itl_p50, itl_p90, itl_p95, itl_p99, itl_min, itl_max = stats(all_itls)
    e2e_mean, e2e_p50, e2e_p90, e2e_p95, e2e_p99, e2e_min, e2e_max = stats(e2es)

    summary = BenchmarkSummary(
        timestamp=datetime.now(UTC).isoformat(),
        model_name=model_name or "active-server-model",
        endpoint=endpoint,
        concurrency=concurrency,
        request_rate=request_rate,
        total_requests=total_prompts,
        successful_requests=success_count,
        failed_requests=fail_count,
        total_duration_s=round(total_duration, 4),
        request_throughput_rps=round(req_tps, 2),
        output_token_throughput_tps=round(out_tok_tps, 2),
        total_token_throughput_tps=round(tot_tok_tps, 2),
        ttft_mean_s=round(ttft_mean, 4),
        ttft_median_s=round(ttft_p50, 4),
        ttft_p90_s=round(ttft_p90, 4),
        ttft_p95_s=round(ttft_p95, 4),
        ttft_p99_s=round(ttft_p99, 4),
        ttft_min_s=round(ttft_min, 4),
        ttft_max_s=round(ttft_max, 4),
        itl_mean_ms=round(itl_mean, 2),
        itl_median_ms=round(itl_p50, 2),
        itl_p90_ms=round(itl_p90, 2),
        itl_p95_ms=round(itl_p95, 2),
        itl_p99_ms=round(itl_p99, 2),
        itl_min_ms=round(itl_min, 2),
        itl_max_ms=round(itl_max, 2),
        e2e_mean_s=round(e2e_mean, 4),
        e2e_median_s=round(e2e_p50, 4),
        e2e_p90_s=round(e2e_p90, 4),
        e2e_p95_s=round(e2e_p95, 4),
        e2e_p99_s=round(e2e_p99, 4),
        e2e_min_s=round(e2e_min, 4),
        e2e_max_s=round(e2e_max, 4),
        gpu_available=gpu_stats["gpu_available"],
        gpu_device_name=gpu_stats["gpu_device_name"],
        gpu_total_mb=gpu_stats["gpu_total_mb"],
        gpu_baseline_mb=gpu_stats["gpu_baseline_mb"],
        gpu_peak_mb=gpu_stats["gpu_peak_mb"],
        gpu_delta_mb=gpu_stats["gpu_delta_mb"],
        gpu_per_stream_mb=gpu_stats["gpu_per_stream_mb"],
        gpu_residual_mb=gpu_stats["gpu_residual_mb"],
    )

    return summary, results


def print_summary_table(summary: BenchmarkSummary):
    """Pretty-print benchmark and GPU memory results to standard output."""
    border = "=" * 70
    sub_border = "-" * 70
    print("\n" + border)
    print(f"       INFERENCE ENGINE BENCHMARK REPORT: Concurrency = {summary.concurrency}")
    print(border)
    print(f"Endpoint:              {summary.endpoint}")
    print(f"Total Requests:        {summary.total_requests} (Success: {summary.successful_requests}, Failed: {summary.failed_requests})")
    print(f"Elapsed Time:          {summary.total_duration_s:.2f} s")
    print(sub_border)
    print("THROUGHPUT METRICS:")
    print(f"  • Request Throughput:      {summary.request_throughput_rps:.2f} req/s")
    print(f"  • Output Token Throughput: {summary.output_token_throughput_tps:.2f} tokens/s")
    print(f"  • Total Token Throughput:  {summary.total_token_throughput_tps:.2f} tokens/s")
    print(sub_border)
    print("TIME TO FIRST TOKEN (TTFT) [seconds]:")
    print(f"  • Mean:    {summary.ttft_mean_s:.4f} s")
    print(f"  • Median:  {summary.ttft_median_s:.4f} s (p50)")
    print(f"  • p90:     {summary.ttft_p90_s:.4f} s")
    print(f"  • p99:     {summary.ttft_p99_s:.4f} s")
    print(f"  • Min/Max: {summary.ttft_min_s:.4f} s / {summary.ttft_max_s:.4f} s")
    print(sub_border)
    print("INTER-TOKEN LATENCY (ITL / TPOT) [milliseconds]:")
    print(f"  • Mean:    {summary.itl_mean_ms:.2f} ms")
    print(f"  • Median:  {summary.itl_median_ms:.2f} ms (p50)")
    print(f"  • p90:     {summary.itl_p90_ms:.2f} ms")
    print(f"  • p99:     {summary.itl_p99_ms:.2f} ms")
    print(f"  • Min/Max: {summary.itl_min_ms:.2f} ms / {summary.itl_max_ms:.2f} ms")
    print(sub_border)
    print("END-TO-END LATENCY (E2E) [seconds]:")
    print(f"  • Mean:    {summary.e2e_mean_s:.4f} s")
    print(f"  • Median:  {summary.e2e_median_s:.4f} s (p50)")
    print(f"  • p90:     {summary.e2e_p90_s:.4f} s")
    print(f"  • p99:     {summary.e2e_p99_s:.4f} s")

    if summary.gpu_available:
        print(sub_border)
        print(f"GPU MEMORY PROFILING ({summary.gpu_device_name}):")
        print(f"  • Total VRAM:        {summary.gpu_total_mb:,.1f} MB")
        print(f"  • Baseline VRAM:     {summary.gpu_baseline_mb:,.1f} MB (Idle before load)")
        print(f"  • Peak VRAM:         {summary.gpu_peak_mb:,.1f} MB (Maximum during batch)")
        print(f"  • Delta VRAM:        {summary.gpu_delta_mb:,.1f} MB (Batch allocation spike)")
        print(f"  • VRAM per Stream:   {summary.gpu_per_stream_mb:,.2f} MB / stream")
        print(f"  • Residual VRAM:     {summary.gpu_residual_mb:+,.1f} MB (Offset after finish)")

    print(border + "\n")


def main():
    parser = argparse.ArgumentParser(description="LLM Serving Benchmark & Memory Profiling Client")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="Base URL of inference server")
    parser.add_argument("--endpoint", type=str, default="/v1/chat/completions", choices=["/v1/chat/completions", "/v1/completions"])
    parser.add_argument("--api-key", type=str, required=True, help="Bearer token for API auth (required)")
    parser.add_argument("--dataset", type=str, default="benchmarks/datasets/sample_prompts.json", help="Path to prompts JSON file (defaults to benchmarks/datasets/sample_prompts.json)")
    parser.add_argument("--num-prompts", type=int, default=20, help="Total number of prompts to fire")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent clients (ignored if --sweep-concurrency is set)")
    parser.add_argument("--sweep-concurrency", type=str, default=None, help="Comma-separated list of concurrency levels to sweep (e.g. '1,4,16,64')")
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Poisson arrival rate (req/s). Default inf = max concurrency")
    parser.add_argument("--no-stream", action="store_true", help="Disable SSE streaming (measures only E2E latency)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Directory to save benchmark JSON output")
    parser.add_argument("--model", type=str, default=None, help="Override target model name")

    args = parser.parse_args()
    api_key = args.api_key
    stream = not args.no_stream

    # Setup prompt workload from dataset file
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[Error] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    prompts = load_dataset_prompts(dataset_path, args.num_prompts)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concurrency_list = [args.concurrency]
    if args.sweep_concurrency:
        concurrency_list = [int(c.strip()) for c in args.sweep_concurrency.split(",") if c.strip()]

    all_results: List[Dict[str, Any]] = []

    for c in concurrency_list:
        print(f"\n[Benchmarking] Running with concurrency = {c}, total requests = {args.num_prompts}, stream = {stream}...")
        summary, raw_metrics = asyncio.run(
            run_benchmark(
                base_url=args.host,
                endpoint=args.endpoint,
                api_key=api_key,
                prompts=prompts,
                concurrency=c,
                request_rate=args.request_rate,
                stream=stream,
                model_name=args.model,
            )
        )
        print_summary_table(summary)
        all_results.append({
            "summary": asdict(summary),
            "raw_metrics": [asdict(m) for m in raw_metrics],
        })

    # Dump all benchmark results into a single unified file at the end
    ts_slug = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_results_{ts_slug}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] All benchmark results written to {out_file}")


if __name__ == "__main__":
    main()
