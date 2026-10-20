from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .analyzer import analyze, load_model
from .scraper import ScrapeError, fetch_reviews

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    load_model()  # load once at startup
    yield


app = FastAPI(title="Flipkart Review Analyzer", lifespan=lifespan)


class UrlIn(BaseModel):
    url: str
    max_pages: int = Field(15, ge=1, le=50)  # ~10 reviews per page


class TextIn(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=5000)
    product: str = "Pasted reviews"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/analyze")
def analyze_url(body: UrlIn):  # sync def => runs in a worker thread, doesn't block the server
    try:
        name, reviews = fetch_reviews(body.url, body.max_pages)
        return analyze(reviews, name)
    except ScrapeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/analyze-text")
def analyze_text(body: TextIn):
    try:
        return analyze(body.reviews, body.product)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
