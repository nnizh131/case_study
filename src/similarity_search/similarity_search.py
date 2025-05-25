import torch
import clip
from PIL import Image
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import requests
from io import BytesIO
from pathlib import Path
import ssl
import urllib.request

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for TransactionImageSearch."""

    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "transaction_image_search"
    clip_model: str = "ViT-B/32"
    ssl_verify: bool = False
    top_k_default: int = 5


class TransactionImageSearch:
    def __init__(self, config: Config | None = None):
        """
        Initialize CLIP model and Qdrant client for transaction and image search

        Args:
            config: Optional Config dataclass for configuration
        """
        self.config = config or Config()

        if not self.config.ssl_verify:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            def install_ssl_handler():
                https_handler = urllib.request.HTTPSHandler(context=ssl_context)
                opener = urllib.request.build_opener(https_handler)
                urllib.request.install_opener(opener)

            install_ssl_handler()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        try:
            self.model, self.preprocess = clip.load(
                self.config.clip_model, device=self.device
            )
        except Exception as e:
            try:
                import tempfile

                temp_dir = tempfile.mkdtemp()
                self.model, self.preprocess = clip.load(
                    self.config.clip_model, device=self.device, download_root=temp_dir
                )
            except Exception as e2:
                raise RuntimeError(
                    "Could not load CLIP model. Please check your internet connection and SSL certificates."
                )

        self.collection_name = self.config.collection_name
        try:
            self.client = QdrantClient(
                host=self.config.qdrant_url, port=self.config.qdrant_port
            )
            self.client.get_collections()
            self._create_collection()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Qdrant: {e}")

    def _create_collection(self):
        """
        Ensure a fresh Qdrant collection for CLIP embeddings.

        If a collection with the same name already exists, it will be dropped
        and recreated so the index starts from a clean state.
        """
        # CLIP ViT‑B/32 vectors are 512‑dimensional
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    def _upsert_points(self, points):
        """Upsert points to Qdrant"""
        self.client.upsert(collection_name=self.collection_name, points=points)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0]

    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image using CLIP"""
        try:
            if image_path.startswith(("http://", "https://")):
                response = requests.get(image_path, verify=False)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

            with torch.no_grad():
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                return image_features.cpu().numpy()[0]

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def create_transaction_text(self, row: pd.Series) -> str:
        """
        Create a rich text description from transaction data for CLIP encoding

        Args:
            row: Pandas Series containing transaction data

        Returns:
            Formatted text description
        """
        parts = []

        parts.append(f"${float(row['amount']):.2f} purchase")
        parts.append(f"in {row['category']}")

        if pd.notna(row["subcategory"]) and row["subcategory"]:
            parts.append(f"for {row['subcategory']}")

        if pd.notna(row["description"]) and row["description"]:
            parts.append(f"described as {row['description']}")

        if pd.notna(row["vendor"]) and row["vendor"]:
            parts.append(f"from {row['vendor']}")

        if "user_id" in row and pd.notna(row["user_id"]):
            parts.append(f"by user {row['user_id']}")

        if pd.notna(row["date"]):
            ts = pd.to_datetime(row["date"])
            parts.append(
                f"on {ts.day} {ts.strftime('%B')} {ts.year}"
            )  # e.g., "on 24 May 2025"

        return " ".join(parts)

    def add_transactions_from_df(self, df: pd.DataFrame):
        points = []
        for idx, row in df.iterrows():
            text_description = self.create_transaction_text(row)

            embedding = self.encode_text(text_description)

            date_parts = {}
            if pd.notna(row["date"]):
                ts = pd.to_datetime(row["date"])
                date_parts = {
                    "date_iso": ts.strftime("%Y-%m-%d"),
                    "year": int(ts.year),
                    "month": int(ts.month),
                    "day": int(ts.day),
                }

            payload = {
                "type": "transaction",
                "transaction_id": row["transaction_id"],
                "user_id": row["user_id"],
                **date_parts,
                "amount": float(row["amount"]),
                "category": row["category"],
                "subcategory": row.get("subcategory", ""),
                "description": row.get("description", ""),
                "vendor": row.get("vendor", ""),
                "text_representation": text_description,
            }

            points.append(
                PointStruct(
                    id=hash(row["transaction_id"])
                    % (10**9),  # Use transaction_id hash as point ID
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        self._upsert_points(points)

    def add_images_from_directory(self, image_dir: str, metadata_csv: str = None):
        """
        Add images from a directory to the search index

        Args:
            image_dir: Directory containing images
            metadata_csv: Optional CSV file with image metadata (filename, description, etc.)
        """
        image_dir = Path(image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        metadata_df = None
        if metadata_csv and os.path.exists(metadata_csv):
            metadata_df = pd.read_csv(metadata_csv)
            metadata_df = (
                metadata_df.set_index("filename")
                if "filename" in metadata_df.columns
                else metadata_df
            )

        points = []
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() in image_extensions:
                try:
                    embedding = self.encode_image(str(image_path))
                    payload = {
                        "type": "image",
                        "filename": image_path.name,
                        "path": str(image_path),
                        "size": image_path.stat().st_size,
                    }
                    if metadata_df is not None and image_path.name in metadata_df.index:
                        metadata_row = metadata_df.loc[image_path.name]
                        for col in metadata_df.columns:
                            if pd.notna(metadata_row[col]):
                                payload[col] = metadata_row[col]
                    points.append(
                        PointStruct(
                            id=hash(str(image_path)) % (10**9),
                            vector=embedding.tolist(),
                            payload=payload,
                        )
                    )
                except Exception:
                    pass
        if points:
            self._upsert_points(points)

    def search_transactions(
        self,
        query: str,
        top_k: int | None = None,
        amount_range: tuple[float, float] | None = None,
        category: str | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Search for transactions using text query with optional filters

        Args:
            query: Text query
            top_k: Number of results to return
            amount_range: Optional (min, max) amount filter
            category: Optional category filter

        Returns:
            List of (payload, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_default
        query_embedding = self.encode_text(query)
        filter_conditions = {
            "must": [{"key": "type", "match": {"value": "transaction"}}]
        }
        if amount_range:
            filter_conditions["must"].append(
                {
                    "key": "amount",
                    "range": {"gte": amount_range[0], "lte": amount_range[1]},
                }
            )
        if category:
            filter_conditions["must"].append(
                {"key": "category", "match": {"value": category}}
            )
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=filter_conditions
                if len(filter_conditions["must"]) > 1
                else {"must": [{"key": "type", "match": {"value": "transaction"}}]},
                limit=top_k,
            )
            return [(result.payload, result.score) for result in search_results]
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}")

    def search_images(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[dict, float]]:
        """
        Search for images using text query

        Args:
            query: Text query describing the image
            top_k: Number of results to return

        Returns:
            List of (payload, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_default
        query_embedding = self.encode_text(query)
        filter_conditions = {"must": [{"key": "type", "match": {"value": "image"}}]}
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=filter_conditions,
                limit=top_k,
            )
            return [(result.payload, result.score) for result in search_results]
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}")

    def search_by_image(
        self, image_path: str, content_type: str = "both", top_k: int | None = None
    ) -> list[tuple[dict, float]]:
        """
        Search using an image query

        Args:
            image_path: Path to the query image
            content_type: "transactions", "images", or "both"
            top_k: Number of results to return

        Returns:
            List of (payload, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_default
        query_embedding = self.encode_image(image_path)
        filter_conditions = None
        if content_type == "transactions":
            filter_conditions = {
                "must": [{"key": "type", "match": {"value": "transaction"}}]
            }
        elif content_type == "images":
            filter_conditions = {"must": [{"key": "type", "match": {"value": "image"}}]}
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=filter_conditions,
                limit=top_k,
            )
            return [(result.payload, result.score) for result in search_results]
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}")

    def find_related_content(
        self, transaction_id: str, top_k: int | None = None
    ) -> list[tuple[dict, float]]:
        """
        Find images and other transactions related to a specific transaction

        Args:
            transaction_id: ID of the reference transaction
            top_k: Number of results to return

        Returns:
            List of (payload, score) tuples
        """
        if top_k is None:
            top_k = self.config.top_k_default
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "transaction_id",
                            "match": {"value": transaction_id},
                        }
                    ]
                },
                limit=1,
            )
            if not search_result[0]:
                return []
            transaction_point = search_result[0][0]
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=transaction_point.vector,
                limit=top_k + 1,
            )
            results = []
            for result in search_results:
                if result.payload.get("transaction_id") != transaction_id:
                    results.append((result.payload, result.score))
            return results[:top_k]
        except Exception as e:
            raise RuntimeError(f"Qdrant find_related_content failed: {e}")

    def get_statistics(self) -> dict[str, int | str | float]:
        """Get statistics about the indexed data"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            transaction_count = len(
                self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [{"key": "type", "match": {"value": "transaction"}}]
                    },
                    limit=10000,
                )[0]
            )

            image_count = len(
                self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter={
                        "must": [{"key": "type", "match": {"value": "image"}}]
                    },
                    limit=10000,
                )[0]
            )

            return {
                "total_points": collection_info.points_count,
                "transactions": transaction_count,
                "images": image_count,
                "vector_size": collection_info.config.params.vectors.size,
                "storage": "qdrant",
            }
        except Exception as e:
            raise RuntimeError(f"Error getting Qdrant stats: {e}")
