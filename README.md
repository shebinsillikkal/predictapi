# PredictAPI — ML Inference Service

> Serves multiple ML models in production through one clean API — 5ms average response time, 3,000+ concurrent requests.

## The Problem
Needed to serve several different ML models through one API without blocking on slow inference. FastAPI async handles it perfectly — one service, multiple models, no queuing issues.

## Features
- ⚡ Async inference — 3,000+ simultaneous requests
- 📦 Multi-model registry — hot-swap versions without downtime
- 📖 Auto-generated Swagger docs
- 🚦 Rate limiting + usage metering
- 📊 Per-model latency and error tracking

## Stack
```
FastAPI | Python | Pydantic | Uvicorn | Redis | Prometheus | Docker
```

**Built by Shebin S Illikkal** — Shebinsillikkal@gmail.com
