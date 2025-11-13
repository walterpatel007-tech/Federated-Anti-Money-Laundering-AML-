"""Streamlit dashboard for exploring AML transaction trends."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from aml_data_pipeline.transform import engineer_features

st.set_page_config(page_title="Federated AML Dashboard", layout="wide")

st.title("Federated AML Analytics")
st.write(
    "This dashboard provides a quick overview of transaction patterns across the"
    " participating banks. Upload a CSV to begin exploring."
)

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
    data = engineer_features(data)
    st.subheader("Summary Statistics")
    st.dataframe(data.describe(include="all"))

    if "is_suspicious" in data.columns:
        st.subheader("Suspicious vs. normal")
        suspicious_counts = data["is_suspicious"].value_counts().rename(
            index={0: "Normal", 1: "Suspicious"}
        )
        st.bar_chart(suspicious_counts)

    st.subheader("Transaction Amount Distribution")
    st.line_chart(data.groupby(data["timestamp"].dt.date)["amount"].sum())
else:
    sample_path = Path("data/processed/sample_transactions.csv")
    if sample_path.exists():
        st.info("Loading sample dataset from data/processed/sample_transactions.csv")
        data = pd.read_csv(sample_path, parse_dates=["timestamp"])
        data = engineer_features(data)
        st.dataframe(data.head())
    else:
        st.warning("Upload a dataset to see analytics.")
