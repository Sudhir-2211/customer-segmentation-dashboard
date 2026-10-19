from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class CustomerSegmenter:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca = PCA(n_components=3, random_state=random_state)

    def fit_predict(self, rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Fits KMeans on scaled RFM values and returns enriched DataFrame with cluster IDs,
        PCA coordinates, and the overall silhouette score.
        """
        features = ["recency", "frequency", "monetary"]
        X = rfm_df[features].copy()

        # Handle positive log scaling to tame skewness
        X_log = np.log1p(X)
        X_scaled = self.scaler.fit_transform(X_log)

        clusters = self.kmeans.fit_predict(X_scaled)
        rfm_enriched = rfm_df.copy()
        rfm_enriched["cluster"] = clusters

        score = silhouette_score(X_scaled, clusters) if len(rfm_df) > self.n_clusters else 0.0

        # Compute 3D PCA coordinates for interactive plotting
        pca_coords = self.pca.fit_transform(X_scaled)
        rfm_enriched["pca_x"] = pca_coords[:, 0].round(3)
        rfm_enriched["pca_y"] = pca_coords[:, 1].round(3)
        rfm_enriched["pca_z"] = pca_coords[:, 2].round(3)

        return rfm_enriched, round(float(score), 3)

    @staticmethod
    def evaluate_cluster_range(rfm_df: pd.DataFrame, min_k: int = 2, max_k: int = 7) -> List[Dict[str, Any]]:
        """Evaluates inertia (Elbow) and silhouette scores across a range of k."""
        features = ["recency", "frequency", "monetary"]
        X_scaled = StandardScaler().fit_transform(np.log1p(rfm_df[features]))

        results = []
        for k in range(min_k, min(max_k + 1, len(rfm_df))):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            results.append({
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette": round(float(sil), 3),
            })
        return results
