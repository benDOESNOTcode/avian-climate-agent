# 🦅 Avian Climate Intelligence

A serverless, multi-modal AI agent that analyzes, visualizes, and forecasts bird population dynamics by synthesizing historical climate and sighting data — all through natural language.

---

## Overview

Ecological data is inherently multi-dimensional: millions of quantitative sighting records alongside rich qualitative species profiles. Traditionally, extracting insight from this data requires domain experts to manually coordinate SQL databases, GIS tools, and charting libraries.

**Avian Climate Intelligence** unifies these workflows into a single conversational agent. Ask questions in plain English — the system routes each query to the right analytical engine, executes it, and returns an answer, chart, map, or forecast.

---

## Features

- **Natural Language Querying** — Ask about species behavior, habitat, or population trends without writing any code
- **Semantic Search (RAG)** — Vector similarity retrieval over 1,000+ encyclopedic species profiles
- **Big-Data SQL Analytics** — Zero-copy DuckDB queries over cloud-hosted Parquet files via `httpfs`
- **Ecological Forecasting** — Regression-based population shift predictions parameterized by species and temperature delta
- **Autonomous Chart Generation** — A self-correcting LangGraph state machine that writes, executes, and debugs its own Plotly code
- **Geospatial WebGL Visualization** — 60fps Pydeck maps rendering 20-year species migration trajectories with RGBA trail effects

---

## Architecture

The system's core is a **multi-agent routing engine** built on LangChain and LangGraph, using `Llama-3.3-70b-versatile` (via Groq) as the central reasoning engine. Incoming queries are classified and dispatched to one of four specialized pathways:

```
User Query
    │
    ▼
┌─────────────────────────────┐
│     Master Routing Agent    │  ← Llama-3.3-70b via Groq
└─────────────┬───────────────┘
              │
     ┌────────┼────────┬──────────────┐
     ▼        ▼        ▼              ▼
 VECTOR      SQL    PREDICT        CHART
 (Qdrant)  (DuckDB) (Regression) (LangGraph)
```

| Route | Trigger | What It Does |
|---|---|---|
| `VECTOR` | Qualitative questions | Embeds query → Qdrant top-K retrieval → LLM synthesis |
| `SQL` | Quantitative aggregations | LLM writes DuckDB SQL from a strict data dictionary → executes → returns results |
| `PREDICT` | Population forecasting | Extracts species + temperature delta → regression model → LLM narrates output |
| `CHART` | Visualization requests | LangGraph cyclical state machine generates, executes, and self-corrects Plotly code |

### The LangGraph Charting State Machine

The charting agent is designed to eliminate the brittleness of zero-shot code generation:

1. LLM generates a SQL query to fetch chart data
2. DuckDB executes it; exact `df.dtypes` and sample rows are extracted
3. Verified schema is fed back to the LLM, which writes a `plotly.express` script
4. Script executes in a sandboxed `exec()` environment
5. On any failure (SQL error, Plotly exception), the traceback is appended to the prompt and the loop retries — up to a configurable retry limit

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Inference | Groq (`Llama-3.3-70b-versatile`) |
| Agent Orchestration | LangChain, LangGraph |
| Analytical Database | DuckDB + `httpfs` extension |
| Vector Store | Qdrant |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| Geospatial Visualization | Pydeck (WebGL) |
| Chart Generation | Plotly Express |
| Data Layer | Apache Parquet (Hugging Face Hub) |
| UI | Streamlit |
| Data Processing | Pandas, PyArrow |

---

## Datasets

### Quantitative Data Lake
- **Source:** Remote `.parquet` files hosted on [Hugging Face Hub](https://huggingface.co/datasets/notBEn/avian-climate-intelligence)
- **Scale:** Millions of records; queried in-memory via DuckDB `httpfs` — no local download required
- **Key fields:** `species`, `total_historical_sightings`, `avg_flock_size`, `latitude`, `longitude`, climate delta features

### Qualitative Species Profiles
- **Source:** Encyclopedic textual profiles for 1,000+ avian species
- **Content:** Habitat, foraging behavior, nesting phenology, geographic range
- **Storage:** Embedded with `all-MiniLM-L6-v2` and indexed in Qdrant

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

```bash
git clone https://github.com/<your-username>/avian-climate-intelligence.git
cd avian-climate-intelligence
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Run Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Ingest Species Profiles

```bash
python ingest.py
```

### Launch the App

```bash
streamlit run app.py
```

---

## Project Structure

```
avian-climate-intelligence/
├── app.py                  # Streamlit UI entry point
├── agent/
│   ├── router.py           # Master routing agent
│   ├── vector_chain.py     # RAG pipeline (Qdrant)
│   ├── sql_chain.py        # Text-to-SQL engine (DuckDB)
│   ├── predict_chain.py    # Forecasting pipeline
│   └── chart_agent.py      # LangGraph charting state machine
├── data/
│   └── ingest.py           # Embedding + Qdrant ingestion script
├── viz/
│   └── geo_map.py          # Pydeck geospatial visualization
├── requirements.txt
└── .env.example
```

---

## Future Work

- **Multi-Tenant Deployment** — Replace local Qdrant and `exec()` sandbox with isolated Docker containers for secure cloud deployment
- **Conversational Memory** — LangChain memory buffer to maintain context across multi-turn conversations
- **Real-Time Ingestion** — Connect Parquet pipeline to live bird-sighting APIs (e.g., [eBird](https://ebird.org/)) for real-time anomaly detection

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
