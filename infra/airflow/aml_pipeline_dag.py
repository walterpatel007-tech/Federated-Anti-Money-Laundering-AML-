"""Airflow DAG sketch for the AML data + training pipeline."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from aml_data_pipeline.ingest import load_raw_transactions, write_processed_data
from aml_data_pipeline.partition import partition_by_bank
from aml_data_pipeline.transform import engineer_features


def extract(**context):  # pragma: no cover - orchestration stub
    load_raw_transactions("data/raw")


def transform(**context):  # pragma: no cover - orchestration stub
    df = load_raw_transactions("data/raw")
    df = engineer_features(df)
    write_processed_data(df, "data/processed", "transactions_processed.csv")


def partition(**context):  # pragma: no cover - orchestration stub
    df = load_raw_transactions("data/raw")
    df = engineer_features(df)
    partition_by_bank(df, "data/bank_splits", "bank_id")


def trigger_training(**context):  # pragma: no cover - orchestration stub
    # Placeholder for CLI invocation or API call to start federated training
    print("Trigger federated training")


default_args = {"owner": "aml", "start_date": datetime(2023, 1, 1)}

with DAG(
    dag_id="aml_federated_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    extract_task = PythonOperator(task_id="extract_raw", python_callable=extract)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    partition_task = PythonOperator(task_id="partition", python_callable=partition)
    train_task = PythonOperator(task_id="trigger_training", python_callable=trigger_training)

    extract_task >> transform_task >> partition_task >> train_task
