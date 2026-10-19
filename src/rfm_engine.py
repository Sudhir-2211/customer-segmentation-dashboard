from datetime import datetime
from typing import Optional
import pandas as pd


class RFMEngine:
    @staticmethod
    def calculate_rfm(
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary metrics from raw transactional logs.
        Columns required: ['customer_id', 'order_date', 'amount']
        """
        if df.empty:
            return pd.DataFrame(columns=["customer_id", "recency", "frequency", "monetary", "aov"])

        clean_df = df.copy()
        clean_df["order_date"] = pd.to_datetime(clean_df["order_date"])

        if reference_date is None:
            # Use max order date + 1 day as baseline reference
            reference_date = clean_df["order_date"].max() + pd.Timedelta(days=1)

        rfm = clean_df.groupby("customer_id").agg({
            "order_date": lambda dates: (reference_date - dates.max()).days,
            "transaction_id": "count",
            "amount": "sum",
        }).reset_index()

        rfm.rename(columns={
            "order_date": "recency",
            "transaction_id": "frequency",
            "amount": "monetary",
        }, inplace=True)

        rfm["monetary"] = rfm["monetary"].round(2)
        rfm["aov"] = (rfm["monetary"] / rfm["frequency"]).round(2)

        return rfm
