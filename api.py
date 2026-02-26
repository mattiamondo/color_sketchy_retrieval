"""
FastAPI backend for the text-to-image search engine.

Exposes a /search endpoint backed by SigLIP2SearchEngine and serves
image files as static assets so the browser can render results.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from search_cli import DATASET_CONFIGS, SigLIP2SearchEngine

BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "data" / "color" / "images"

config = DATASET_CONFIGS["color"]


class SearchResultItem(BaseModel):
    rank: int
    score: float
    image_url: str
    name: str


class SearchResponse(BaseModel):
    query: str
    elapsed_ms: float
    results: list[SearchResultItem]


engine: SigLIP2SearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SigLIP2SearchEngine(
        embeddings_path=str(config["embeddings"]),
        metadata_path=str(config["metadata"]),
        image_key=config["image_key"],
        image_dir=str(config["image_dir"]),
    )
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=20, ge=1, le=100),
) -> SearchResponse:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    results, elapsed_ms = engine.search(query.strip(), top_k=top_k)

    items = [
        SearchResultItem(
            rank=r.rank,
            score=r.score,
            # Serve the image via the /images static mount using only the filename.
            image_url=f"/images/{Path(r.path).name}",
            name=r.metadata.get(config["caption_key"], ""),
        )
        for r in results
    ]

    return SearchResponse(query=query, elapsed_ms=elapsed_ms, results=items)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
