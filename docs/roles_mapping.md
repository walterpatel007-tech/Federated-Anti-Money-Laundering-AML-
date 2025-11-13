# Role Mapping

| Role               | Primary Assets                                    | Responsibilities |
|--------------------|----------------------------------------------------|------------------|
| Data Engineer      | `aml_data_pipeline/`, `infra/airflow/`             | Build ingestion pipelines, validation, feature engineering, orchestration |
| Data Analyst       | `notebooks/`, `analytics_dashboard/`               | Explore raw and processed data, craft dashboards and EDA reports |
| Data Scientist     | `aml_federated_training/lightning_module.py`, `notebooks/03_model_eval_federated.ipynb` | Design model architectures, run experiments, analyze metrics |
| ML Engineer        | `aml_federated_training/client_flower.py`, `server_flower.py`, `train_federated.py` | Coordinate Flower training loops, manage configuration, monitor runs |
| Software Engineer  | `aml_serving_api/`, `infra/docker/Dockerfile.api`  | Package and serve the global model, ensure API reliability |
| Infrastructure Ops | `infra/docker-compose.yml`, `infra/docker/`        | Containerize services and manage local deployment |

The repo structure keeps ownership explicit, allowing each role to iterate
independently while sharing artifacts through clear interfaces.
