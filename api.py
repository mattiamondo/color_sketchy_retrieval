"""
FastAPI backend for the text-to-image search engine.

Exposes a /search endpoint backed by SigLIP2SearchEngine and serves
image files as static assets so the browser can render results.

In production (no Vite dev server), FastAPI also serves the pre-built
React frontend from frontend/dist/ — mount it last so API routes take
priority over the catch-all SPA handler.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from search_cli import DATASET_CONFIGS, SigLIP2SearchEngine

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "data" / "color" / "images"
SKETCHY_IMAGE_DIR = BASE_DIR / "data" / "sketchy_test" / "images"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"

config = DATASET_CONFIGS["color"]

ROCCHIO_ALPHA = 0.8
ROCCHIO_BETA = 0.2


class SearchResultItem(BaseModel):
    rank: int
    score: float
    image_url: str
    name: str
    embedding_index: int


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    results: list[SearchResultItem]


class RefineRequest(BaseModel):
    query: str
    mode: str
    positive_indices: list[int]
    negative_indices: list[int]


class RefineResponse(BaseModel):
    results: list[SearchResultItem]
    elapsed_ms: float


engine: SigLIP2SearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SigLIP2SearchEngine(
        embeddings_path=str(config["embeddings"]),
        metadata_path=str(config["metadata"]),
        image_key=config["image_key"],
        image_dir=str(config["image_dir"]),
        text_embeddings_path=str(config["text_embeddings"]),
    )
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.mount("/sketchy-images", StaticFiles(directory=SKETCHY_IMAGE_DIR), name="sketchy-images")


def _make_item(idx: int, rank: int, score: float) -> SearchResultItem:
    return SearchResultItem(
        rank=rank,
        score=score,
        image_url=f"/images/{Path(engine.image_paths[idx]).name}",
        name=engine.metadata[idx].get(config["caption_key"], "") if idx < len(engine.metadata) else "",
        embedding_index=idx,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=20, ge=1, le=100),
    mode: str = Query(default="image", pattern="^(image|text)$"),
) -> SearchResponse:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    results, elapsed_ms = engine.search(query.strip(), top_k=top_k, mode=mode)

    items = [_make_item(r.embedding_index, r.rank, r.score) for r in results]

    return SearchResponse(query=query, elapsed_ms=elapsed_ms, results=items)


@app.post("/refine", response_model=RefineResponse)
def refine(req: RefineRequest) -> RefineResponse:
    import time
    t0 = time.perf_counter()

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if not req.positive_indices and not req.negative_indices:
        raise HTTPException(status_code=400, detail="At least one positive or negative index required")

    emb_matrix = (
        engine.text_embeddings
        if req.mode == "text" and engine.text_embeddings is not None
        else engine.image_embeddings
    )

    q = engine.encode_text(req.query.strip()).squeeze()  # (D,)

    if req.positive_indices:
        q = q + ROCCHIO_ALPHA * emb_matrix[req.positive_indices].mean(axis=0)
    if req.negative_indices:
        q = q - ROCCHIO_BETA * emb_matrix[req.negative_indices].mean(axis=0)

    q = q / np.linalg.norm(q)

    scores = np.dot(emb_matrix, q).squeeze()
    top_indices = np.argsort(scores)[::-1][:20]

    items = [_make_item(int(idx), rank, float(scores[idx])) for rank, idx in enumerate(top_indices, start=1)]

    return RefineResponse(results=items, elapsed_ms=(time.perf_counter() - t0) * 1000)


# Serve the pre-built React app if the dist folder exists.
# Must be mounted after all API routes so /search and /images take priority.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=False)
