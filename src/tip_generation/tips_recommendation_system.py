import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    """Configuration for EASE recommender."""

    l2_reg: float = 300.0
    top_k_default: int = 3
    seed: int = 42


class EASERecommender:
    """
    EASE (Embarrassingly Shallow Autoencoder) (Steck 2019) recommender for tax deduction categories.
    https://arxiv.org/pdf/1905.03375
    """

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the recommender with optional configuration.

        Args:
            config: Config instance; if None, uses default settings.
        """
        self._config = config or Config()
        self._user2idx: dict[str, int] = {}
        self._cat2idx: dict[str, int] = {}
        self._X: sp.csr_matrix | None = None
        self._W: np.ndarray | None = None
        self._pop_norm: np.ndarray | None = None
        self._impact_vec: np.ndarray | None = None

    def load_data(
        self,
        transactions_df: pd.DataFrame,
        users_df: pd.DataFrame,
        tax_df: pd.DataFrame,
    ) -> None:
        """
        Load and preprocess user, transaction, and tax data.

        Args:
            transactions_df: Transactions with columns ['user_id','category','amount'].
            users_df: User DataFrame (only for consistent ordering).
            tax_df: Tax filings with columns ['user_id','refund_amount'].
        """
        self._user2idx = {u: i for i, u in enumerate(users_df["user_id"].unique())}
        agg = (
            transactions_df.dropna(subset=["user_id", "category", "amount"])
            .groupby(["user_id", "category"], as_index=False)["amount"]
            .sum()
        )
        agg["value"] = np.log1p(agg["amount"])
        cats = agg["category"].unique()
        self._cat2idx = {c: i for i, c in enumerate(cats)}

        rows = agg["user_id"].map(self._user2idx)
        cols = agg["category"].map(self._cat2idx)
        data = agg["value"]
        self._X = sp.coo_matrix(
            (data, (rows, cols)), shape=(len(self._user2idx), len(self._cat2idx))
        ).tocsr()

        pop_vec = (self._X > 0).sum(axis=0).A1
        self._pop_norm = pop_vec / (pop_vec.max() or 1)
        user_refund = (
            tax_df.groupby("user_id", as_index=False)["refund_amount"]
            .mean()
            .rename(columns={"refund_amount": "avg_refund"})
        )
        impact_per_cat = []
        for cat in cats:
            users_with = agg.loc[agg["category"] == cat, "user_id"].unique()
            users_without = np.setdiff1d(list(self._user2idx.keys()), users_with)
            ref_with = user_refund[user_refund.user_id.isin(users_with)][
                "avg_refund"
            ].mean()
            ref_without = user_refund[user_refund.user_id.isin(users_without)][
                "avg_refund"
            ].mean()
            impact_per_cat.append(
                0.0
                if np.isnan(ref_with) or np.isnan(ref_without)
                else ref_with - ref_without
            )
        impact_vec = np.array(impact_per_cat, dtype=np.float32)
        min_l, max_l = impact_vec.min(), impact_vec.max()
        self._norm_impact = (impact_vec - min_l) / (max_l - min_l + 1e-6)
        self._max_impact = max_l

    def train(self) -> None:
        """
        Train EASE weight matrix W from the loaded data.
        """
        if self._X is None:
            raise RuntimeError("Data not loaded; call load_data() first")
        G = (self._X.T @ self._X).toarray().astype(np.float64)
        diag = np.arange(G.shape[0])
        G[diag, diag] += self._config.l2_reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0)
        self._W = B.astype(np.float32)
        logger.info("EASE training finished.")

    def recommend(self, user_id: str, k: int | None = None) -> list[str]:
        """
        Recommend top-k categories the user is missing.

        Args:
            user_id: Target user identifier.
            k: Number of recommendations; uses config default if None.

        Returns:
            List of category names.
        """
        if self._W is None or self._X is None:
            raise RuntimeError("Model not trained; call train() first")
        k = k or self._config.top_k_default
        if user_id not in self._user2idx:
            raise ValueError(f"user_id {user_id} not found")
        u = self._user2idx[user_id]
        user_row = self._X[u]
        base = (user_row @ self._W).ravel()
        penalty = self._max_impact - self._norm_impact
        scores = base - penalty
        scores[user_row.indices] = -np.inf
        idxs = np.argsort(-scores)[:k]
        cats = list(self._cat2idx.keys())
        return [cats[i] for i in idxs]

    def evaluate(self, k: int | None = None) -> dict[str, Any]:
        """
        Leave-one-out evaluation: HR@k, MAP@k.

        Args:
            k: Cut-off for evaluation metrics.

        Returns:
            dict with keys 'hr','map'.
        """
        if self._W is None or self._X is None:
            raise RuntimeError("Model not trained; call train() first")
        k = k or self._config.top_k_default
        rng = np.random.default_rng(self._config.seed)
        hits = ap = n = 0
        for u in range(self._X.shape[0]):
            items = self._X[u].indices
            if len(items) < 2:
                continue
            test = rng.choice(items)
            train = items[items != test]
            row = sp.csr_matrix(
                (np.ones_like(train), ([0] * len(train), train)),
                shape=(1, self._X.shape[1]),
            )
            base = (row @ self._W).ravel()
            penalty = self._max_impact - self._norm_impact
            scores = base - penalty
            scores[train] = -np.inf
            topk = np.argpartition(-scores, k)[:k]
            topk = topk[np.argsort(-scores[topk])]
            hit = int(test in topk)
            hits += hit
            if hit:
                rank = np.where(topk == test)[0][0] + 1
                ap += 1.0 / rank
            n += 1
        return {
            f"HR@{k}": hits / n if n else 0.0,
            f"MAP@{k}": ap / n if n else 0.0,
        }
