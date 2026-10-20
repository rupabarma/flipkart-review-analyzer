# Flipkart Review Analyzer (FastAPI)

## Run
    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn app.main:app --reload
Open http://127.0.0.1:8000

## Retrain the model (also needed if you see a scikit-learn pickle/version error)
    python train_sentiment.py --csv flipkart_product.csv --out model

## Files
- app/main.py      FastAPI routes (`/`, `/api/analyze`, `/api/analyze-text`)
- app/scraper.py   Fetches reviews from a Flipkart link (selectors may need updating)
- app/analyzer.py  Preprocessing, sentiment, feature extraction, worth-buying score
- app/static/index.html  The website

## Notes
- Flipkart may block scraping or change its page layout; if so, update `BODY_SELECTORS` in scraper.py.
- Worth score = average probability a review is positive (shrunk toward 50% for few reviews).
  Flipkart reviews skew positive, so most products score high; compare products against each other.

## If review fetching fails
Install the optional browser mode: `pip install playwright && playwright install chromium`, restart the server.
Otherwise open `debug_last_response.html` (saved in the project folder) to see what Flipkart returned, and update `BODY_SELECTORS` in app/scraper.py.
