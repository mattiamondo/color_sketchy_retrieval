#!/usr/bin/env python3
"""
Text-to-image semantic search engine using pre-computed SigLIP-2 embeddings.

Usage:
    python search_cli.py --query "red dress" --dataset color
    python search_cli.py --query "blue shoes" --embeddings path/to/embeddings.npy --metadata path/to/metadata.json
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os

import numpy as np
import torch

# Suppress HuggingFace HTTP and progress noise — only errors surface.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from transformers import AutoModel, AutoProcessor

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
EMBEDS_BASE = BASE_DIR / "embeddings"
DATA_BASE = BASE_DIR / "data"

SIGLIP2_CHECKPOINT = "google/siglip2-so400m-patch14-384"

# Must match the value used in compute_embeddings_and_stats_siglip2.py
# to keep query and image embeddings in the same distribution.
MAX_TEXT_LENGTH = 64

DATASET_CONFIGS: dict[str, dict] = {
    "sketchy_test": {
        "embeddings": EMBEDS_BASE / "sketchy_test" / "siglip2_image.npy",
        "metadata": DATA_BASE / "sketchy_test" / "metadata.json",
        "image_key": "original_filename",
        "caption_key": "original_caption",
    },
    "color": {
        "embeddings": EMBEDS_BASE / "color" / "siglip2_image.npy",
        "metadata": DATA_BASE / "color" / "metadata.json",
        "image_key": "image",
        "image_dir": DATA_BASE / "color",
        "caption_key": "name",
    },
}


@dataclass
class SearchResult:
    rank: int
    score: float
    path: str
    metadata: dict


class SigLIP2SearchEngine:
    """Text-to-image retrieval engine backed by pre-computed SigLIP-2 image embeddings."""

    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        image_key: str,
        image_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            embeddings_path: Path to .npy file of L2-normalized image embeddings, shape (N, D).
            metadata_path: Path to JSON file with per-image metadata.
            image_key: Metadata field that holds the image file path.
            image_dir: Base directory used to resolve relative image paths.
        """
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("Device: %s", self.device)

        self.image_embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info("Loaded embeddings: shape %s", self.image_embeddings.shape)

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        self.image_paths = self._resolve_image_paths(image_key, image_dir)
        logger.info("Indexed %d images", len(self.image_paths))

        if len(self.image_paths) != self.image_embeddings.shape[0]:
            logger.warning(
                "Metadata/embedding count mismatch: %d vs %d",
                len(self.image_paths),
                self.image_embeddings.shape[0],
            )

        t0 = time.perf_counter()
        logger.info("Loading SigLIP-2: %s", SIGLIP2_CHECKPOINT)
        self.model = AutoModel.from_pretrained(SIGLIP2_CHECKPOINT).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(SIGLIP2_CHECKPOINT, use_fast=True)
        logger.info("Model ready (%.1fs)", time.perf_counter() - t0)

    def _resolve_image_paths(self, image_key: str, image_dir: Optional[str]) -> list[str]:
        raw = [entry.get(image_key, "") for entry in self.metadata]
        if not image_dir:
            return raw
        base = Path(image_dir)
        return [
            str(base / p) if p and not Path(p).is_absolute() else p
            for p in raw
        ]

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text query into an L2-normalized embedding vector."""
        with torch.no_grad():
            inputs = self.processor(
                text=[query],
                padding="max_length",
                max_length=MAX_TEXT_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            features = self.model.get_text_features(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )
            # L2 normalization makes dot product equivalent to cosine similarity.
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def search(self, query: str, top_k: int = 5) -> tuple[list[SearchResult], float]:
        """
        Retrieve the top-k images most similar to the text query.

        Returns results and query latency in milliseconds.
        """
        t0 = time.perf_counter()

        query_emb = self.encode_text(query)
        # Both sides are L2-normalized, so dot product equals cosine similarity.
        scores = np.dot(self.image_embeddings, query_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            SearchResult(
                rank=rank,
                score=float(scores[idx]),
                path=self.image_paths[idx],
                metadata=self.metadata[idx] if idx < len(self.metadata) else {},
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

        return results, (time.perf_counter() - t0) * 1000


def print_results(
    results: list[SearchResult],
    query: str,
    elapsed_ms: float,
    caption_key: str = "",
) -> None:
    print(f"\n{'=' * 60}")
    print(f'Query: "{query}"  ({elapsed_ms:.1f} ms)')
    print("=" * 60)
    for r in results:
        print(f"\n[{r.rank}] score={r.score:.4f}")
        print(f"    {r.path}")
        if caption_key and (caption := r.metadata.get(caption_key, "")):
            print(f"    {caption[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-to-image semantic search with SigLIP-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python search_cli.py --query "red dress" --dataset color
  python search_cli.py --query "blue shoes" --embeddings ./embeds.npy --metadata ./meta.json
        """,
    )
    parser.add_argument("--query", "-q", required=True, help="Search query text")
    parser.add_argument("--dataset", "-d", choices=list(DATASET_CONFIGS), help="Predefined dataset name")
    parser.add_argument("--embeddings", "-e", help="Path to image embeddings .npy file")
    parser.add_argument("--metadata", "-m", help="Path to metadata JSON file")
    parser.add_argument("--image-key", "-k", default="original_filename", help="Metadata field for image path")
    parser.add_argument("--top-k", "-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Enter interactive query loop")
    args = parser.parse_args()

    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        embeddings_path = str(config["embeddings"])
        metadata_path = str(config["metadata"])
        image_key = config["image_key"]
        image_dir = str(config["image_dir"]) if "image_dir" in config else None
        caption_key = config.get("caption_key", "")
    elif args.embeddings and args.metadata:
        embeddings_path = args.embeddings
        metadata_path = args.metadata
        image_key = args.image_key
        image_dir = None
        caption_key = ""
    else:
        parser.error("Provide --dataset or both --embeddings and --metadata")

    for label, path in [("embeddings", embeddings_path), ("metadata", metadata_path)]:
        if not Path(path).exists():
            logger.error("%s file not found: %s", label, path)
            sys.exit(1)

    engine = SigLIP2SearchEngine(embeddings_path, metadata_path, image_key, image_dir)

    if args.interactive:
        print("[Interactive mode — type 'exit/quit/q' to quit]")
        while True:
            try:
                query = input("\nQuery> ").strip()
                if query.lower() in {"exit", "quit", "q"}:
                    break
                if query:
                    results, elapsed_ms = engine.search(query, args.top_k)
                    print_results(results, query, elapsed_ms, caption_key)
            except KeyboardInterrupt:
                break
    else:
        results, elapsed_ms = engine.search(args.query, args.top_k)
        print_results(results, args.query, elapsed_ms, caption_key)


if __name__ == "__main__":
    main()
