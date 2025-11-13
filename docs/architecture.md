# Architecture Overview

This project simulates three independent financial institutions collaborating on
an Anti–Money Laundering (AML) detection system. The implementation focuses on
modularity so individual roles can develop their portion of the stack while
adhering to shared contracts.

```
Data Pipeline -> Bank Partitions -> Federated Training -> Global Model -> Serving API
                              \-> Analytics Dashboard
```

1. **Data Pipeline (`aml_data_pipeline/`)**
   - Ingests AML transaction CSVs
   - Validates schema with Pydantic models
   - Applies feature engineering and normalization steps
   - Saves per-bank splits for federated clients

2. **Federated Training (`aml_federated_training/`)**
   - Wraps a PyTorch Lightning model in Flower clients
   - Coordinates cross-silo training rounds
   - Produces a global model checkpoint for downstream consumers

3. **Serving (`aml_serving_api/`)**
   - FastAPI service that loads the global model
   - Accepts transaction payloads and returns suspiciousness scores

4. **Analytics (`analytics_dashboard/`)**
   - Streamlit app to explore transaction distributions and suspicious flags

5. **Infrastructure (`infra/`)**
   - Dockerfiles and compose definition to run server, clients, and API services

The architecture supports experimentation and incremental improvements without
requiring access to sensitive raw data outside each simulated bank.
