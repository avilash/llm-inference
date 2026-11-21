# LLM Continuous Batching Inference Engine

## Overview

A high-throughput, low-latency LLM/SLM inference server built from scratch in Python. It implements dynamic iteration-level continuous batching, an asynchronous background execution engine, and two-tier authentication with OpenAI-compatible API endpoints.

## Features

- **Continuous (Iteration-Level) Batching**: Dynamically schedules incoming requests and evicts completed sequences at each token decode step to eliminate padding waste.
- **OpenAI-Compatible API**: Drop-in compatibility for `/v1/chat/completions`, `/v1/completions`, and `/v1/models`, supporting Server-Sent Events (SSE) token streaming (`stream: true`).
- **Two-Tier Authentication & Rate Limiting**: Sub-millisecond auth lookups via Redis session cache backed by persistent PostgreSQL storage.
- **Asynchronous Architecture**: Non-blocking FastAPI request routing paired with an `asyncio` background step loop and threaded PyTorch model execution.

## How to Run

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Docker & Docker Compose

### 2. Start Infrastructure
Start PostgreSQL 16 and Redis 7:
```bash
docker compose up -d
```

### 3. Setup Environment & Dependencies
```bash
cp .env.example .env
uv sync
```

### 4. Start the Server
```bash
uv run python main.py
```
The API server initializes tables, preloads the model (default: `Qwen/Qwen2.5-0.5B-Instruct`), and listens on `http://localhost:8000`.

### 5. Generate an API Key
Create an API key to access protected endpoints:
```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "dev-key", "tenant_id": "dev"}'
```
*(Or use the local key in `api_key.json` if available).*

---

## Benchmarking & Memory Profiling

The engine includes a client-side benchmarking and GPU profiling harness in `benchmarks/` that measures end-to-end performance without modifying server code.

### 1. Run a Standard Benchmark
Benchmark streaming latency and throughput with 4 concurrent clients:
```bash
uv run python benchmarks/benchmark_serving.py --api-key "<YOUR_API_KEY>" --concurrency 4 --num-prompts 20
```

### 2. Run a Concurrency Scaling Sweep (1, 4, 16, 64)
Profile the unpaged KV cache baseline across multiple concurrency levels to measure throughput scaling and VRAM exhaustion:
```bash
uv run python benchmarks/benchmark_serving.py --api-key "<YOUR_API_KEY>" --sweep-concurrency 1,4,16,64 --num-prompts 64
```

### 3. Visualize Comparative Metrics
Generate comparative Markdown tables, ASCII charts in the terminal, and optional PNG plots:
```bash
uv run python benchmarks/plot_results.py
```

### Metrics Collected
* **Latency**: TTFT (Time To First Token), ITL (Inter-Token Latency / TPOT), and E2E request latency distributions (Mean, Median p50, p90, p95, p99).
* **Throughput**: Request throughput (req/s), output token throughput, and total token throughput (tokens/s).
* **GPU Memory**: Baseline VRAM, peak allocated VRAM, batch allocation spike ($\Delta\text{VRAM}$), VRAM consumed per concurrent stream, and residual memory leaks via NVML.


