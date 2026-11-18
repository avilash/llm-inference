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

