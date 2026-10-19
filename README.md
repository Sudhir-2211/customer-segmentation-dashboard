# Customer Segmentation Dashboard

An unsupervised machine learning dashboard for customer lifecycle intelligence and RFM (Recency, Frequency, Monetary) segmentation built with **Python**, **Scikit-learn**, **Plotly**, and **Streamlit**.

---

## Features

- **RFM Feature Engineering**: Converts raw transactional timestamp logs into customer-level behavioral metrics:
  - `Recency`: Days elapsed since the last completed transaction.
  - `Frequency`: Total transaction volume per customer.
  - `Monetary`: Cumulative historical spend.
  - `AOV`: Average Order Value ($Monetary / Frequency$).
- **K-Means Clustering & Dimensionality Reduction**:
  - StandardScaler with log transformation to handle right-skewed spending distributions.
  - Optimal cluster discovery using Elbow method (inertia) and Silhouette Score analysis.
  - 3D Principal Component Analysis (PCA) for visual cluster separation.
- **Automated Persona Mapping & Marketing Playbooks**:
  - `Champions`: VIP perks, referral programs, and early access.
  - `Loyal Customers`: Upsell opportunities, value bundles, loyalty accelerators.
  - `At-Risk High Spenders`: High prior spend with rising dormancy; urgent win-back campaigns.
  - `Hibernating / Lost`: Low historical spend and high dormancy; automated re-activation or cost-conscious suppression.
- **Interactive Plotly Visualization**: 3D interactive cluster scatter plot, cohort summary cards, and customer lookup utility.

---

## Project Structure

```text
customer-segmentation-dashboard/
├── sample_data/
│   ├── generate_transactions.py    # Synthetic customer transaction seeder
│   └── customer_transactions.csv   # Synthetic transaction dataset
├── src/
│   ├── rfm_engine.py               # RFM calculation and feature engineering
│   ├── segmentation.py             # K-Means clustering, PCA, and Silhouette evaluation
│   └── persona_mapper.py           # Persona labeling and strategic recommendations
├── tests/
│   ├── test_rfm_engine.py          # Unit tests for RFM transformation
│   ├── test_segmentation.py        # Clustering convergence & evaluation tests
│   └── test_persona_mapper.py      # Persona mapping logic tests
├── app.py                          # Streamlit web application
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+

### 2. Installation

```bash
cd customer-segmentation-dashboard
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

python -m pip install -r requirements.txt
```

### 3. Running the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Automated Tests

Run the PyTest test suite:

```bash
python -m pytest tests -v
```

Tests cover:
- RFM calculation accuracy and edge-case handling on empty data
- K-Means fitting and output shape consistency
- PCA 3-component projection validation
- Silhouette score computation within expected ranges
- Persona assignment and strategy lookup

---

## Security Considerations

1. **Personally Identifiable Information (PII)**: Transaction logs often contain customer emails and physical addresses. Ensure data uploaded to the dashboard has PII masked or anonymized before ingestion.
2. **Access Control**: Segmentation insights reveal revenue patterns and should be restricted to authenticated marketing and operations teams.

---

## Known Limitations

- **Spherical Cluster Assumption**: K-Means assumes spherical cluster geometry; for highly irregular non-convex distributions, density-based clustering (e.g. DBSCAN / HDBSCAN) may be required.
- **Static Temporal Snapshot**: Clustering reflects historical cumulative spend; does not predict forward-looking Customer Lifetime Value (CLV).
