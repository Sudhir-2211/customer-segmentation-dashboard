from typing import Dict, Any
import pandas as pd


PERSONA_STRATEGIES = {
    "Champions": {
        "description": "High value, highly frequent buyers who recently transacted.",
        "strategy": "Reward with early product access, VIP concierge, and referral incentives.",
        "color": "#10b981",
    },
    "Loyal Customers": {
        "description": "Consistent repeat purchasers with reliable cadence and healthy order values.",
        "strategy": "Upsell premium tiers, loyalty point accelerators, and cross-sell related categories.",
        "color": "#3b82f6",
    },
    "At-Risk High Spenders": {
        "description": "Historically valuable accounts who have not ordered recently.",
        "strategy": "Urgent win-back campaigns, personal account executive outreach, and custom renewal discounts.",
        "color": "#f59e0b",
    },
    "Hibernating / Lost": {
        "description": "Infrequent buyers with high recency dormancy and low historical spend.",
        "strategy": "Automated re-engagement drip sequence, discount clearance offers, or prune from active ad spend.",
        "color": "#ef4444",
    },
}


def label_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns business personas to numeric cluster labels based on RFM profile medians."""
    cluster_stats = clustered_df.groupby("cluster").agg({
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean",
    }).reset_index()

    # Score clusters: higher frequency & monetary = positive, higher recency = negative
    cluster_stats["composite_score"] = (
        cluster_stats["frequency"] * 0.35 +
        cluster_stats["monetary"] / cluster_stats["monetary"].max() * 50 -
        cluster_stats["recency"] * 0.25
    )

    sorted_clusters = cluster_stats.sort_values(by="composite_score", ascending=False)["cluster"].tolist()

    persona_order = ["Champions", "Loyal Customers", "At-Risk High Spenders", "Hibernating / Lost"]
    mapping = {}
    for idx, c_id in enumerate(sorted_clusters):
        mapping[c_id] = persona_order[idx % len(persona_order)]

    df = clustered_df.copy()
    df["persona"] = df["cluster"].map(mapping)
    return df
