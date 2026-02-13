# Agent Observability Platform

A backend service for AI task execution with built-in execution tracing using Ollama for local LLM inference.

The platform records hierarchical execution traces (with spans, decisions, errors, and logs) in PostgreSQL.  
All observability data is stored as traces in the database.

---

## Features

- AI task execution (summarize, analyze, classify, extract, translate)
- Hierarchical execution tracing (trace + spans)
- LLM-assisted decision tracking
- Error context attached to traces
- Ollama integration (local models)
- PostgreSQL persistence
- pgAdmin included for DB inspection

---

## Architecture

**Services (Docker Compose):**

- `app` – Flask backend API
- `postgres` – Main database (stores traces)
- `ollama` – Local LLM runtime
- `pgadmin` – Database UI

All execution traces are stored in PostgreSQL.  

---

# Setup Guide

## 1. Requirements

- Docker
- Docker Compose

---

## 2. Environment Configuration

Navigate to the `deploy` folder:

```bash
cd deploy
```

Copy the example environment file if needed:

```bash
cp env.example .env
```

(Adjust values if required.)

---

## 3. Start the Platform

From inside the `deploy` folder:

```bash
docker compose up --build
```

This will start:

- PostgreSQL on `localhost:5432`
- Ollama on `localhost:11434`
- pgAdmin on `http://localhost:5050`
- Backend API on `http://localhost:5000`

---

## 4. Ollama Model

The container automatically:

- Starts `ollama serve`
- Pulls the model defined in .env

Default model used by the app:
```
deepseek-coder
```

---

## 5. Accessing Services

### Backend API
```
http://localhost:5000
```

### pgAdmin
```
http://localhost:5050
Email: admin@observability.com
Password: admin
```

PostgreSQL connection details inside pgAdmin:

- Host: `postgres`
- Username: `postgres`
- Password: `postgres`
- Database: `observability`

---

## Data Model (High-Level)

### Trace
- trace_id
- start_time
- end_time
- metadata

### Span
- span_id
- parent_span_id
- operation_name
- attributes
- start_time
- end_time

### Events / Logs
- trace_id
- span_id
- level
- message
- payload

All execution activity is captured through traces and spans stored in PostgreSQL.

---

## API Overview

### Execute Task
```
POST /api/tasks/{type}
```

Example types:
- summarize
- analyze
- classify
- extract
- translate

Returns a `trace_id`.

---

### Get Trace
```
GET /api/traces/{trace_id}
```

Returns full trace with spans and events.

---

### List Models
```
GET /api/models
```

---

## Logs

Application logs are mounted to:

```
observability_project/logs/
```

---

## Stopping the Platform

From `deploy` folder:

```bash
docker compose down
```

To remove volumes:

```bash
docker compose down -v
```

---
