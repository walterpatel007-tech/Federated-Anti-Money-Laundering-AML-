# Federated Anti–Money Laundering (AML)

This repository contains a reference implementation of a federated learning
pipeline for Anti–Money Laundering (AML) detection. The goal is to simulate a
collaboration between three financial institutions that train a shared global
model without exchanging raw transaction data.

The project demonstrates how multiple data-focused roles can work together in a
single code base:

- **Data Engineer** – builds ingestion, validation, and feature engineering
  pipelines that prepare AML transaction records for analysis.
- **Data Analyst** – performs exploratory data analysis and designs a dashboard
  to monitor trends in suspicious activity.
- **Data Scientist** – experiments with modeling approaches and evaluates
  results across local and federated training strategies.
- **ML Engineer** – orchestrates Flower-based cross-silo training and handles
  experiment configuration.
- **Software Engineer** – serves the trained global model behind a FastAPI
  inference service.

The repository is structured so that each role has an associated package or
workspace. The accompanying documentation highlights how the pieces connect from
raw data through model serving.

## Getting Started

Create a Python virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation

Place AML transaction CSVs (for example, from the open-source
[AMLSim](https://github.com/IBM/AMLSim) generator) into `data/raw/`. Then run
`aml_data_pipeline.ingest.load_raw_transactions` to load and validate the data.

```python
from aml_data_pipeline.ingest import load_raw_transactions
from aml_data_pipeline.partition import partition_by_bank

raw_df = load_raw_transactions("data/raw")
partition_by_bank(raw_df, output_dir="data/bank_splits", bank_column="bank_id")
```

See `config/data_config.yaml` for additional configuration options such as the
transaction schema and partition definitions.

### Federated Training

A full Flower simulation can be launched with:

```bash
python -m aml_federated_training.train_federated \
    --data-config config/data_config.yaml \
    --model-config config/model_config.yaml \
    --federated-config config/federated_config.yaml
```

This command spins up a Flower server and three simulated clients (one for each
bank) on the same machine. Each client trains a PyTorch Lightning model on its
local split, and the global weights are aggregated via FedAvg.

### Serving the Global Model

After federated training, export the resulting state dict to
`artifacts/global_model.ckpt`. The FastAPI service in `aml_serving_api` can load
this checkpoint and respond to `/predict_transaction` requests with suspicious
activity scores.

```bash
uvicorn aml_serving_api.main:app --reload
```

### Dashboards and Notebooks

Exploratory notebooks live in `notebooks/`. The Streamlit dashboard found in
`analytics_dashboard/app.py` gives a high-level view of transaction trends for
business stakeholders.

## Repository Layout

The repository layout mirrors the project brief and keeps responsibilities
clearly scoped.

```
.
├── aml_data_pipeline/        # Data ingestion, validation, feature engineering
├── aml_federated_training/   # PyTorch Lightning model + Flower training stack
├── aml_serving_api/          # FastAPI inference service for the global model
├── analytics_dashboard/      # Streamlit dashboard for AML analytics
├── config/                   # YAML configuration files
├── docs/                     # High-level documentation and diagrams
├── infra/                    # Docker and orchestration assets
├── notebooks/                # Jupyter notebooks for analysis and experiments
└── requirements.txt          # Project dependencies
```

Each package contains extensive inline documentation to guide further
development. The initial code favors readability and composability so the
project can evolve into a full end-to-end demonstration of federated AML
modeling.
