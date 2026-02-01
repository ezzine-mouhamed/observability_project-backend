# Agent Observability and Execution Tracing System with Ollama

A production-ready backend service that demonstrates comprehensive observability and execution tracing for AI-powered tasks using Ollama for local LLM inference. The system captures execution traces, decision contexts, failure signals, and performance metrics.

## Features

- **Ollama Integration**: Local LLM inference with multiple model support
- **Execution Tracing**: Hierarchical trace capture with parent-child relationships
- **Decision Context**: Records why specific execution paths were chosen using LLM-assisted decision making
- **Structured Logging**: JSON-formatted logs for machine readability
- **Performance Metrics**: Aggregated metrics for monitoring and optimization
- **Error Context**: Rich error information with execution context
- **Embedding Support**: Text embeddings using Ollama models
- **Model Management**: API endpoints for Ollama model management

## Architecture Overview

### Core Components

1. **Task Execution Service**: Manages AI task execution with built-in observability
2. **Ollama LLM Client**: Local LLM inference with fallback mechanisms
3. **Decision Engine**: LLM-assisted decision making with tracing
4. **Observability Layer**: Captures traces, metrics, and events throughout execution
5. **Persistence Layer**: Stores execution data for analysis and debugging

### Supported Task Types

- `summarize`: Text summarization
- `analyze`: Data analysis and insights
- `classify`: Text classification
- `extract`: Information extraction
- `translate`: Text translation

## Prerequisites

### System Requirements

- Python 3.11+
- Ollama installed and running
- SQLite (for development) or PostgreSQL (for production)

### Ollama Setup

1. **Install Ollama**:
