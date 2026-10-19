import pytest
import pandas as pd
from datetime import datetime
from src.rfm_engine import RFMEngine


def test_calculate_rfm_standard_flow():
    data = [
        {"transaction_id": 1, "customer_id": "C-1", "order_date": "2026-09-01", "amount": 100.0},
        {"transaction_id": 2, "customer_id": "C-1", "order_date": "2026-09-15", "amount": 200.0},
        {"transaction_id": 3, "customer_id": "C-2", "order_date": "2026-08-01", "amount": 50.0},
    ]
    df = pd.DataFrame(data)
    ref = datetime(2026, 10, 1)

    rfm = RFMEngine.calculate_rfm(df, reference_date=ref)
    assert len(rfm) == 2

    c1 = rfm[rfm["customer_id"] == "C-1"].iloc[0]
    assert c1["frequency"] == 2
    assert c1["monetary"] == 300.0
    assert c1["aov"] == 150.0
    assert c1["recency"] == 16  # 2026-10-01 - 2026-09-15

    c2 = rfm[rfm["customer_id"] == "C-2"].iloc[0]
    assert c2["frequency"] == 1
    assert c2["monetary"] == 50.0


def test_calculate_rfm_empty_input():
    empty_df = pd.DataFrame()
    rfm = RFMEngine.calculate_rfm(empty_df)
    assert rfm.empty
    assert "recency" in rfm.columns
