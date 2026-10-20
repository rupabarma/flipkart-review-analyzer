"""Preprocessing, sentiment prediction, feature (aspect) extraction and worth-buying score."""
import re
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "sentiment_model.joblib"
_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def clean_text(s: str) -> str:  # must match train_sentiment.clean_text
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\x00-\x7f]", " ", s)
    s = re.sub(r"read more", " ", s)
    s = re.sub(r"([!?.])\1+", r"\1", s)
    s = re.sub(r"(.)\1{3,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()


# Feature -> keyword prefixes. Edit freely to add product-specific features.
ASPECTS = {
    "Quality & build": ["quality", "build", "material", "sturdy", "solid", "finish", "plastic", "flimsy", "durab", "long lasting"],
    "Price & value": ["price", "value for money", "worth", "money", "cost", "budget", "expensive", "affordable", "deal"],
    "Performance": ["performance", "speed", "fast", "slow", "lag", "hang", "power", "cooling", "heating", "efficien", "smooth", "work"],
    "Battery & charging": ["battery", "charg", "backup"],
    "Sound": ["sound", "audio", "bass", "volume", "speaker", "mic", "noise", "music"],
    "Display": ["display", "screen", "brightness", "resolution"],
    "Camera": ["camera", "photo", "video", "picture"],
    "Design & looks": ["design", "look", "style", "stylish", "colour", "color", "sleek", "compact", "weight", "size"],
    "Comfort & fit": ["comfort", "fit", "soft", "fabric", "grip", "wear"],
    "Delivery & packaging": ["deliver", "packag", "packing", "shipping", "courier", "arriv", "box"],
    "Service & warranty": ["service", "support", "warranty", "seller", "replace", "refund", "return", "installation"],
}
_ASPECT_RE = {a: re.compile(r"\b(?:%s)" % "|".join(map(re.escape, w))) for a, w in ASPECTS.items()}
_CLAUSE_SPLIT = re.compile(r"[.!?;\n,]+|\b(?:but|however|although|though|except|while)\b")


def _verdict(score: float) -> str:
    if score >= 80: return "Great buy"
    if score >= 65: return "Good buy"
    if score >= 50: return "Mixed reviews"
    return "Not recommended"


def analyze(reviews, product="Product", pos_cut=0.65, neg_cut=0.35):
    # 1. Preprocess all reviews at once
    rows = [(str(r).strip(), clean_text(str(r))) for r in reviews]
    rows = [r for r in rows if len(r[1]) >= 3]
    if not rows:
        raise ValueError("No usable reviews found.")
    model = load_model()

    # 2. Sentiment of every review in one batch
    probs = model.predict_proba([c for _, c in rows])[:, 1]
    n = len(rows)
    pos = int((probs >= 0.5).sum())

    # 3. Feature-level sentiment: split reviews into clauses, find the feature each
    #    clause talks about, and score the clause with the same model.
    items = []
    for _, c in rows:
        for cl in _CLAUSE_SPLIT.split(c):
            cl = cl.strip()
            if len(cl) >= 4:
                hit = [a for a, rx in _ASPECT_RE.items() if rx.search(cl)]
                if hit:
                    items.append((cl, hit))
    stats = {a: {"pos": 0, "neg": 0, "pos_p": 0.0, "neg_p": 1.0, "pos_ex": None, "neg_ex": None} for a in ASPECTS}
    if items:
        cps = model.predict_proba([cl for cl, _ in items])[:, 1]
        for (cl, hit), p in zip(items, cps):
            for a in hit:
                s = stats[a]
                if p >= pos_cut:
                    s["pos"] += 1
                    if p > s["pos_p"]: s["pos_p"], s["pos_ex"] = p, cl
                elif p <= neg_cut:
                    s["neg"] += 1
                    if p < s["neg_p"]: s["neg_p"], s["neg_ex"] = p, cl

    def top(kind):
        out = [(a, s[kind], s["pos"] + s["neg"], s[kind + "_ex"]) for a, s in stats.items() if s[kind] > 0]
        out.sort(key=lambda x: -x[1])
        return [{"feature": a, "count": c, "share": round(100 * c / t), "example": ex} for a, c, t, ex in out[:6]]

    # 4. Worth-buying score: mean P(positive), shrunk toward 50% when there are few reviews
    worth = 100 * (float(probs.sum()) + 5 * 0.5) / (n + 5)
    return {
        "product": product,
        "total_reviews": n,
        "worth_score": round(worth, 1),
        "verdict": _verdict(worth),
        "confidence": "low" if n < 20 else "medium" if n < 100 else "high",
        "positive_pct": round(100 * pos / n, 1),
        "negative_pct": round(100 * (n - pos) / n, 1),
        "positive_count": pos,
        "negative_count": n - pos,
        "features": {"positive": top("pos"), "negative": top("neg")},
    }
