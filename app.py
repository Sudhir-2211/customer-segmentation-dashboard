import streamlit as st
import pandas as pd
import plotly.express as px
from src.rfm_engine import RFMEngine
from src.segmentation import CustomerSegmenter
from src.persona_mapper import label_clusters, PERSONA_STRATEGIES

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Customer RFM Segmentation & Behavioral Intelligence")
st.markdown("Unsupervised K-Means clustering and persona mapping for targeted retention and lifecycle marketing.")

# Sidebar Configuration
st.sidebar.header("Segmentation Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV (Optional)", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded dataset!")
else:
    raw_df = pd.read_csv("sample_data/customer_transactions.csv")
    st.sidebar.info("Using built-in synthetic dataset (300 customers, ~1,500 transactions).")

n_clusters = st.sidebar.slider("Number of Clusters (k):", min_value=2, max_value=6, value=4)

# Run RFM Engine
rfm_df = RFMEngine.calculate_rfm(raw_df)

# Run Clustering
segmenter = CustomerSegmenter(n_clusters=n_clusters, random_state=42)
clustered_df, silhouette = segmenter.fit_predict(rfm_df)
labeled_df = label_clusters(clustered_df)

# Top KPI Summary
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", f"{len(labeled_df):,}")
col2.metric("Total Revenue", f"${labeled_df['monetary'].sum():,.2f}")
col3.metric("Avg Frequency", f"{labeled_df['frequency'].mean():.1f} orders")
col4.metric("Avg Recency", f"{labeled_df['recency'].mean():.0f} days")
col5.metric("Silhouette Score", f"{silhouette:.3f}")

st.markdown("---")

# Main Visualizations
tab_3d, tab_profiles, tab_eval, tab_lookup = st.tabs([
    "🌐 3D Cluster Visualizer",
    "📋 Persona Profiles & Strategy",
    "📈 Optimal Cluster Evaluation",
    "🔍 Customer Lookup",
])

with tab_3d:
    st.subheader("Interactive 3D Customer Clusters (PCA Projection)")
    st.caption("Dimensionality reduction projecting 3D RFM features into optimal principal components.")

    fig_3d = px.scatter_3d(
        labeled_df,
        x="pca_x",
        y="pca_y",
        z="pca_z",
        color="persona",
        hover_data=["customer_id", "recency", "frequency", "monetary", "aov"],
        title="Customer Persona Distribution",
        color_discrete_map={
            "Champions": "#10b981",
            "Loyal Customers": "#3b82f6",
            "At-Risk High Spenders": "#f59e0b",
            "Hibernating / Lost": "#ef4444",
        },
        opacity=0.85,
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=580)
    st.plotly_chart(fig_3d, use_container_width=True)

with tab_profiles:
    st.subheader("Identified Customer Personas & Retention Playbooks")
    persona_stats = labeled_df.groupby("persona").agg({
        "customer_id": "count",
        "recency": "mean",
        "frequency": "mean",
        "monetary": "mean",
        "aov": "mean",
    }).reset_index()

    cols = st.columns(len(persona_stats))
    for idx, row in persona_stats.iterrows():
        p_name = row["persona"]
        strategy_info = PERSONA_STRATEGIES.get(p_name, {})
        with cols[idx]:
            st.markdown(f"### {p_name}")
            st.markdown(f"**Cohort Size:** {row['customer_id']} customers ({row['customer_id']/len(labeled_df)*100:.1f}%)")
            st.write(f"- **Avg Recency:** {row['recency']:.0f} days")
            st.write(f"- **Avg Frequency:** {row['frequency']:.1f} orders")
            st.write(f"- **Avg Spend:** ${row['monetary']:,.2f}")
            st.write(f"- **Avg Order Value:** ${row['aov']:,.2f}")

            st.info(f"**Strategy:** {strategy_info.get('strategy', 'N/A')}")

with tab_eval:
    st.subheader("Clustering Evaluation (Elbow & Silhouette Analysis)")
    range_eval = CustomerSegmenter.evaluate_cluster_range(rfm_df, min_k=2, max_k=6)
    eval_df = pd.DataFrame(range_eval)

    c1, c2 = st.columns(2)
    with c1:
        fig_inertia = px.line(
            eval_df,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Method (Inertia vs k)",
            color_discrete_sequence=["#6366f1"],
        )
        st.plotly_chart(fig_inertia, use_container_width=True)
    with c2:
        fig_sil = px.bar(
            eval_df,
            x="k",
            y="silhouette",
            title="Silhouette Score by k",
            color="silhouette",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_sil, use_container_width=True)

with tab_lookup:
    st.subheader("Individual Customer Profile Lookup")
    search_id = st.text_input("Enter Customer ID:", value="CUST-0001")
    matched = labeled_df[labeled_df["customer_id"].str.upper() == search_id.strip().upper()]

    if not matched.empty:
        c_row = matched.iloc[0]
        st.success(f"**Persona:** {c_row['persona']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recency", f"{c_row['recency']} days")
        m2.metric("Order Frequency", f"{c_row['frequency']} orders")
        m3.metric("Lifetime Monetary", f"${c_row['monetary']:,.2f}")
        m4.metric("Avg Order Value", f"${c_row['aov']:,.2f}")
    else:
        st.warning(f"Customer '{search_id}' not found.")
