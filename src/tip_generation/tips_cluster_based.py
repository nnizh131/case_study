import pandas as pd
from typing import Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dataclasses import dataclass

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaxTipConfig:
    """Configuration for clustering hyperparameters."""

    random_state: int = 42
    n_init: int = 10
    silhouette_min_k: int = 2
    max_k_ratio: float = 0.5


CONFIG = TaxTipConfig()


class TaxTipGenerator:
    def __init__(self):
        """
        Initialize tip generator
        """
        self.filing_data = None
        self.transaction_data = None
        self.user_data = None
        self.categories = []
        self.user_features = None
        self.clusters = None
        self._scaler = StandardScaler()
        self._label_encoders = {}

    def load_data(
        self,
        filing_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        user_df: pd.DataFrame,
    ) -> None:
        """
        Load input data and build initial user feature matrix.

        Args:
            filing_df (pd.DataFrame): Tax filings (user_id, total_income, total_deductions, refund_amount).
            transaction_df (pd.DataFrame): Transactions (user_id, date, category, amount, ...).
            user_df (pd.DataFrame): User demographics (user_id, occupation_category, age_range, family_status, region).

        Returns:
            None
        """
        self.filing_data = filing_df
        self.transaction_data = transaction_df
        self.user_data = user_df
        self.categories = sorted(self.transaction_data["category"].dropna().unique())
        self._build_user_features()

    def _calculate_user_metrics(self, user_id: str) -> dict[str, Any]:
        """
        Compute filing and spending metrics for a single user.

        Args:
            user_id (str): Identifier of the user.

        Returns:
            dict[str, Any]: Metrics including total_income, deductions, ratios, demographics, transactions_by_category, transaction_count.
        """
        user_filing = self.filing_data[self.filing_data["user_id"] == user_id].iloc[-1]

        user_transactions = self.transaction_data[
            self.transaction_data["user_id"] == user_id
        ]

        user_profile = self.user_data[self.user_data["user_id"] == user_id].iloc[0]

        metrics = {
            "total_income": user_filing["total_income"],
            "total_deductions": user_filing["total_deductions"],
            "deduction_ratio": user_filing["total_deductions"]
            / user_filing["total_income"],
            "refund_amount": user_filing["refund_amount"],
            "refund_ratio": user_filing["refund_amount"] / user_filing["total_income"],
            "occupation": user_profile["occupation_category"],
            "age_range": user_profile["age_range"],
            "family_status": user_profile["family_status"],
            "region": user_profile["region"],
            "transactions_by_category": user_transactions.groupby("category")["amount"]
            .sum()
            .to_dict(),
            "transaction_count": len(user_transactions),
        }
        return metrics

    def _build_user_features(self) -> pd.DataFrame:
        """
        Construct the DataFrame of user-level features for clustering.

        Returns:
            pd.DataFrame: Rows are users, columns are feature values including ratios.
        """
        features_list = []

        for _, user_row in self.user_data.iterrows():
            user_id = user_row["user_id"]

            user_filing = self.filing_data[self.filing_data["user_id"] == user_id]
            if len(user_filing) == 0:
                continue
            user_filing = user_filing.iloc[-1]

            user_transactions = self.transaction_data[
                self.transaction_data["user_id"] == user_id
            ]

            features = {
                "user_id": user_id,
                "income_level": user_filing["total_income"],
                "deduction_ratio": user_filing["total_deductions"]
                / user_filing["total_income"],
                "transaction_frequency": len(user_transactions),
                "avg_transaction_size": user_transactions["amount"].mean()
                if len(user_transactions) > 0
                else 0,
                "spending_diversity": len(user_transactions["category"].unique())
                if len(user_transactions) > 0
                else 0,
                "occupation_category": user_row["occupation_category"],
                "age_range": user_row["age_range"],
                "family_status": user_row["family_status"],
                "region": user_row["region"],
            }

            for cat in self.categories:
                amt = user_transactions.loc[
                    user_transactions["category"] == cat, "amount"
                ].sum()
                features[f"ratio_{cat.replace(' ', '_')}"] = (
                    amt / user_filing["total_income"]
                    if user_filing["total_income"]
                    else 0.0
                )

            features_list.append(features)

        self.user_features = pd.DataFrame(features_list)
        return self.user_features

    def cluster_data(self) -> list[int]:
        """
        Perform clustering on user features using silhouette-optimized KMeans.

        Populates self.user_features['cluster'].

        Returns:
            list[int]: Cluster labels for each user.
        """
        if self.user_features is None or len(self.user_features) < 2:
            return

        clustering_features = self.user_features.copy()

        categorical_cols = [
            "occupation_category",
            "age_range",
            "family_status",
            "region",
        ]
        for col in categorical_cols:
            if col not in self._label_encoders:
                self._label_encoders[col] = LabelEncoder()
            clustering_features[col] = self._label_encoders[col].fit_transform(
                clustering_features[col]
            )

        feature_cols = [
            col
            for col in clustering_features.columns
            if col != "user_id" and not col.startswith("refund")
        ]

        self._clustering_matrix = clustering_features[feature_cols].fillna(0)
        X = self._clustering_matrix

        X_scaled = self._scaler.fit_transform(X)

        n_users = len(X_scaled)
        max_clusters = int(n_users * CONFIG.max_k_ratio)

        best_score = -1
        best_k = 2

        for k in range(CONFIG.silhouette_min_k, max_clusters):
            kmeans = KMeans(
                n_clusters=k,
                random_state=CONFIG.random_state,
                n_init=CONFIG.n_init,
            )
            cluster_labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k

        self.clusters = KMeans(
            n_clusters=best_k,
            random_state=CONFIG.random_state,
            n_init=CONFIG.n_init,
        )
        cluster_labels = self.clusters.fit_predict(X_scaled)

        self.user_features["cluster"] = cluster_labels

        return cluster_labels

    def cluster_summary(self) -> None:
        """
        Log concise summary statistics for each cluster.
        """
        if self.user_features is None:
            return

        for cluster_id, group in self.user_features.groupby("cluster"):
            count = len(group)
            avg_income = group["income_level"].mean()
            avg_ded_ratio = group["deduction_ratio"].mean()
            common_occ = (
                group["occupation_category"].mode().iloc[0]
                if not group.empty
                else "N/A"
            )
            common_age = group["age_range"].mode().iloc[0] if not group.empty else "N/A"
            users = ", ".join(group["user_id"].tolist())
            logger.info(
                f"Cluster {cluster_id}: Users={count}, "
                f"AvgIncome=€{avg_income:,.0f}, "
                f"AvgDedRatio={avg_ded_ratio:.1%}, "
                f"TopOcc={common_occ}, "
                f"AgeRange={common_age}, "
                f"Members=[{users}]"
            )

    def plot_cluster_pca(self):
        """
        Perform PCA on the clustered user features and plot clusters with refund labels.
        """
        if self.user_features is None or self.clusters is None:
            logger.error("No clustering available. Run cluster_data() first.")
            return

        if not hasattr(self, "_clustering_matrix"):
            logger.error("No clustering matrix found. Run cluster_data() first.")
            return
        X_scaled = self._scaler.transform(self._clustering_matrix)

        pca = PCA(n_components=2, random_state=CONFIG.random_state)
        X_pca = pca.fit_transform(X_scaled)

        cluster_labels = self.user_features["cluster"].values
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7
        )
        plt.legend(*scatter.legend_elements(), title="Cluster")
        plt.title("PCA Projection of User Clusters")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        for i, uid in enumerate(self.user_features["user_id"]):
            user_filings = self.filing_data[self.filing_data["user_id"] == uid]
            if not user_filings.empty:
                amount = user_filings["refund_amount"].iloc[-1]
            else:
                amount = 0.0
            plt.text(
                X_pca[i, 0],
                X_pca[i, 1],
                f"€{amount:.0f}",
                fontsize=8,
                alpha=0.8,
            )

        plt.show()

    def _find_cluster_peers(self, user_id: str) -> list[str]:
        """
        Find users in the same cluster as the given user.

        Args:
            user_id (str): User identifier.

        Returns:
            list[str]: Peer user IDs excluding the given user.
        """
        if self.user_features is None:
            return []

        user_row = self.user_features[self.user_features["user_id"] == user_id]
        if len(user_row) == 0:
            return []

        user_cluster = user_row["cluster"].iloc[0]
        cluster_peers = self.user_features[
            (self.user_features["cluster"] == user_cluster)
            & (self.user_features["user_id"] != user_id)
        ]["user_id"].tolist()

        return cluster_peers

    def _calculate_cluster_benchmarks(
        self, cluster_peer_ids: list[str]
    ) -> dict[str, Any]:
        """
        Calculate average metrics for cluster peer user IDs.

        Args:
            cluster_peer_ids (list[str]): List of user IDs in the cluster.

        Returns:
            dict[str, Any]: Benchmark metrics (deduction_ratio, refund_ratio, income, transaction_frequency, per-category ratios).
        """
        if not cluster_peer_ids:
            return {}

        peer_filings = self.filing_data[
            self.filing_data["user_id"].isin(cluster_peer_ids)
        ]
        peer_features = self.user_features[
            self.user_features["user_id"].isin(cluster_peer_ids)
        ]

        if len(peer_filings) == 0:
            return {}

        benchmarks = {
            "avg_deduction_ratio": peer_filings["total_deductions"].sum()
            / peer_filings["total_income"].sum(),
            "avg_refund_ratio": peer_filings["refund_amount"].sum()
            / peer_filings["total_income"].sum(),
            "avg_income": peer_filings["total_income"].mean(),
            "avg_transaction_frequency": peer_features["transaction_frequency"].mean(),
            "cluster_size": len(cluster_peer_ids),
        }

        for cat in self.categories:
            key = f"ratio_{cat.replace(' ', '_')}"
            avg_key = f"avg_{key}"
            benchmarks[avg_key] = peer_features.get(key, pd.Series(0)).mean()

        return benchmarks

    def _generate_cluster_based_tips(
        self, user_metrics: dict[str, Any], cluster_benchmarks: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Generate tips based on spending gaps relative to cluster benchmarks.

        Args:
            user_metrics (dict[str, Any]): Metrics for the user.
            cluster_benchmarks (dict[str, Any]): Benchmark metrics for the user's cluster.

        Returns:
            list[dict[str, Any]]: Tip dicts with title, message, and gap.
        """
        tips = []
        if not cluster_benchmarks:
            return tips

        income = user_metrics["total_income"]

        for cat in self.categories:
            avg_key = f"avg_ratio_{cat.replace(' ', '_')}"
            cluster_avg = cluster_benchmarks.get(avg_key, 0)
            if cluster_avg <= 0:
                continue
            user_amt = user_metrics["transactions_by_category"].get(cat, 0)
            user_ratio = user_amt / income if income else 0

            if user_ratio < cluster_avg:
                gap = cluster_avg - user_ratio
                tips.append(
                    {
                        "type": "cluster_spending_pattern",
                        "title": f"{cat} Expense Gap",
                        "message": (
                            f"Similar users spend {cluster_avg:.1%} of income on {cat.lower()} "
                            f"(vs your {user_ratio:.1%}). You might be missing deductible purchases.\n"
                            f"Action: Review {cat.lower()} expenses"
                        ),
                        "gap": gap,
                    }
                )

        tips.sort(key=lambda t: t.get("gap", 0), reverse=True)
        return tips

    def generate_personalized_tips(
        self, user_id: str, top_k: int = 5
    ) -> dict[str, Any]:
        """
        Generate a personalized tips report for a user.

        Args:
            user_id (str): User identifier.
            top_k (int): Max number of tips to return.

        Returns:
            dict[str, Any]: Report with user_profile, tips, cluster_info, summary.
        """
        try:
            user_metrics = self._calculate_user_metrics(user_id)

            cluster_peers = self._find_cluster_peers(user_id)
            cluster_benchmarks = self._calculate_cluster_benchmarks(cluster_peers)

            all_tips = []
            all_tips.extend(
                self._generate_cluster_based_tips(user_metrics, cluster_benchmarks)
            )

            avg_ref_ratio = cluster_benchmarks.get("avg_refund_ratio", 0.0)
            peer_filings = self.filing_data[
                self.filing_data["user_id"].isin(cluster_peers)
            ]
            if not peer_filings.empty:
                avg_cluster_refund_amount = peer_filings["refund_amount"].mean()
            else:
                avg_cluster_refund_amount = 0.0

            return {
                "user_id": user_id,
                "user_profile": {
                    "occupation": user_metrics["occupation"],
                    "age_range": user_metrics["age_range"],
                    "family_status": user_metrics["family_status"],
                    "income": user_metrics["total_income"],
                    "current_deduction_ratio": user_metrics["deduction_ratio"],
                    "refund_ratio": user_metrics["refund_ratio"],
                    "refund_amount": user_metrics["refund_amount"],
                },
                "tips": all_tips[:top_k],
                "cluster_info": {
                    "cluster_peers": cluster_peers,
                    "cluster_benchmarks": cluster_benchmarks,
                    "avg_cluster_refund_ratio": avg_ref_ratio,
                    "avg_cluster_refund_amount": avg_cluster_refund_amount,
                },
                "summary": {
                    "tips_count": len(all_tips),
                },
            }

        except Exception as e:
            return {"error": f"Error generating tips for user {user_id}: {str(e)}"}
