import pytest
import pandas as pd
from src.rfm_engine import RFMEngine
from src.segmentation import CustomerSegmenter


@pytest.fixture
def sample_rfm():
    transactions = pd.read_csv("sample_data/customer_transactions.csv")
    return RFMEngine.calculate_rfm(transactions)


def test_customer_segmenter_fit_predict(sample_rfm):
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    enriched_df, score = segmenter.fit_predict(sample_rfm)

    assert len(enriched_df) == len(sample_rfm)
    assert "cluster" in enriched_df.columns
    assert "pca_x" in enriched_df.columns
    assert "pca_y" in enriched_df.columns
    assert "pca_z" in enriched_df.columns
    assert enriched_df["cluster"].nunique() == 4
    assert 0.2 <= score <= 0.8  # Realistic silhouette score for RFM data


def test_evaluate_cluster_range(sample_rfm):
    results = CustomerSegmenter.evaluate_cluster_range(sample_rfm, min_k=2, max_k=4)
    assert len(results) == 3
    assert results[0]["k"] == 2
    assert "inertia" in results[0]
    assert "silhouette" in results[0]
