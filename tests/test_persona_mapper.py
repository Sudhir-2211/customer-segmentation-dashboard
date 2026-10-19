import pytest
import pandas as pd
from src.rfm_engine import RFMEngine
from src.segmentation import CustomerSegmenter
from src.persona_mapper import label_clusters, PERSONA_STRATEGIES


def test_label_clusters():
    transactions = pd.read_csv("sample_data/customer_transactions.csv")
    rfm = RFMEngine.calculate_rfm(transactions)
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    enriched_df, _ = segmenter.fit_predict(rfm)

    labeled_df = label_clusters(enriched_df)
    assert "persona" in labeled_df.columns
    unique_personas = labeled_df["persona"].unique()
    assert len(unique_personas) == 4

    # Ensure all assigned personas exist in dictionary
    for p in unique_personas:
        assert p in PERSONA_STRATEGIES
