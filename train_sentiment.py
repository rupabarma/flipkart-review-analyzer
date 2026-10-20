"""
Flipkart review sentiment classifier (Positive / Negative)

Usage:
    python train_sentiment.py --csv flipkart_product.csv --out model/

Labels (derived from star rating):
    4-5 stars -> positive (1)
    1-2 stars -> negative (0)
    3 stars   -> dropped (neutral / ambiguous, would add label noise)

Requirements:
    pip install pandas scikit-learn joblib
"""
import argparse
import os
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_union


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    # The file is not UTF-8; latin-1 reads it without errors.
    df = pd.read_csv(csv_path, encoding="latin-1")

    # A few rows are malformed (product name shifted into the Rate column).
    df = df[df["Rate"].isin(["1", "2", "3", "4", "5"])].copy()
    df["Rate"] = df["Rate"].astype(int)

    # IMPORTANT: use only the review BODY ("Summary" column).
    # The "Review" column holds Flipkart's canned titles ("Awesome", "Fair",
    # "Worth every penny"...). Only ~1,260 distinct values exist and most map to
    # exactly one star rating, so training on them leaks the label and gives
    # fake ~99% accuracy that won't hold up on real review text.
    df["text"] = df["Summary"].fillna("").map(clean_text)
    df = df[df["text"].str.len() > 3]

    # Labels
    df = df[df["Rate"] != 3].copy()
    df["label"] = (df["Rate"] >= 4).astype(int)
    return df[["ProductName", "text", "label"]]


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^\x00-\x7f]", " ", s)          # emoji / mojibake from bad encoding
    s = re.sub(r"read more", " ", s)             # Flipkart UI artefact
    s = re.sub(r"([!?.])\1+", r"\1", s)          # "!!!" -> "!"
    s = re.sub(r"(.)\1{3,}", r"\1\1", s)         # "goooood" -> "good"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------
# 2. Model
# --------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    features = make_union(
        # word 1-2 grams capture "not good", "waste of money", "value for money"
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=300_000),
        # char n-grams make it robust to typos: "gud", "wrost", "awsome"
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                        sublinear_tf=True, max_features=300_000),
    )
    clf = LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


# --------------------------------------------------------------------------
# 3. Train + evaluate
# --------------------------------------------------------------------------
def main(csv_path: str, out_dir: str):
    df = load_data(csv_path)
    print(f"Rows after cleaning: {len(df):,}")
    print(df["label"].value_counts().rename({1: "positive", 0: "negative"}), "\n")

    # The dataset has many near-identical reviews (e.g. "Nice product").
    # Remove exact duplicates so the test set isn't just copies of training rows.
    before = len(df)
    df = df.drop_duplicates(subset=["ProductName", "text"])
    print(f"Removed {before - len(df):,} duplicate reviews -> {len(df):,} remain\n")

    # Split by PRODUCT so the test set contains products the model has never seen.
    # This mimics real use: a user pastes a link to a new product.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["ProductName"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Train: {len(train):,}  |  Test: {len(test):,} (unseen products)\n")

    model = build_pipeline()
    model.fit(train["text"], train["label"])

    pred = model.predict(test["text"])
    print(classification_report(test["label"], pred, target_names=["negative", "positive"], digits=4))
    print("Confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(test["label"], pred))
    print("Macro F1:", round(float(f1_score(test["label"], pred, average="macro")), 4))


    # Retrain on everything for the final model
    model.fit(df["text"], df["label"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sentiment_model.joblib")
    joblib.dump(model, path, compress=3)
    print(f"\nSaved model -> {path}")

    # Quick sanity check
    samples = [
        "Excellent product, totally worth the money",
        "Worst purchase ever, stopped working in 2 days",
        "Not good at all",
        "battery backup is great but the camera is bad",
    ]
    for s, p in zip(samples, model.predict_proba([clean_text(x) for x in samples])[:, 1]):
        print(f"{p:6.2%} positive | {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="flipkart_product.csv")
    ap.add_argument("--out", default="model")
    args = ap.parse_args()
    main(args.csv, args.out)
