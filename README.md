# Industrial Repair Companion

A **Hybrid Diagnostic Assistant** for Siemens SINAMICS Variable Frequency Drive (VFD) systems. Combines official Siemens manual documentation with historical repair log data to provide dual-source diagnostics.

## Features

- **Dual-Source RAG**: Retrieves from both official manuals and 10,500+ historical repair logs
- **Split-Screen UI**: Side-by-side comparison of official procedures vs. field-proven fixes
- **Source Citations**: Every result includes source attribution (manual section or log ID + technician)
- **AI Diagnosis**: LLM-powered diagnostic summaries via Ollama/Llama 3
- **Smart Filters**: Filter by error code, site location, and more

## Quick Start

### 1. Generate Synthetic Data
```bash
python generate_repair_logs.py
```

### 2. Clean & Transform Data
```bash
python clean_data.py
```

### 3. Build Vector Indices
```bash
python ingest.py
```

### 4. Launch the App
```bash
streamlit run app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Ollama + Llama 3 |
| Data Processing | Pandas, Python |
| Deployment | Docker, Docker Compose |

## Project Structure

```
├── data/
│   ├── manuals/              # Siemens VFD manual content
│   ├── repair_logs.csv       # Raw synthetic repair logs
│   └── repair_logs_cleaned.csv # Cleaned + feature-engineered
├── indices/
│   ├── manuals_index/        # FAISS index for manuals
│   └── history_index/        # FAISS index for repair logs
├── generate_repair_logs.py   # Phase 1: Synthetic data generator
├── clean_data.py             # Phase 2: Pandas data pipeline
├── ingest.py                 # Phase 3: Dual FAISS ingestion
├── app.py                    # Phase 4: Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
