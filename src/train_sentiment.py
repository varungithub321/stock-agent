"""Train a simple TensorFlow model to predict positive/neutral/negative sentiment
based on recent news headlines + price movement.

This is a lightweight, demo-focused pipeline:
- Fetches last 7 days of prices and news for provided tickers
- Constructs weak labels using next-day price change (thresholded)
- Trains a small Keras model that combines text and numeric features

Usage:
    python src/train_sentiment.py AAPL,MSFT --epochs 3

Notes:
- This is a demo: labels are weak (based on price movement). For production use,
  you'd want human-labeled data and more robust preprocessing.
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd

import numpy as np

from fetch_prices import get_stock_data
from fetch_news import get_stock_news

# Try TensorFlow, otherwise fall back to scikit-learn
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    USE_TF = False
    print("TensorFlow not available; falling back to scikit-learn classifier.")
    # scikit-learn imports
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib
    from scipy.sparse import hstack


def build_dataset(tickers: List[str], days_back: int = 7, max_per_ticker: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Fetch data and return texts, numeric features, and labels.

    Labels: based on next-day price change
      > +0.5%  -> positive (2)
      < -0.5% -> negative (0)
      else    -> neutral  (1)
    """
    texts = []
    numerics = []
    labels = []
    meta = []  # list of dicts with ticker and date for each sample

    end = datetime.utcnow().date()
    start = end - timedelta(days=days_back + 1)  # plus one to allow next-day label

    for ticker in tickers:
        # fetch price history covering the period
        import yfinance as yf
        hist = yf.Ticker(ticker).history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat())
        if hist.empty:
            continue

        # We'll iterate over available days except the last one (no next day to label)
        dates = list(hist.index.date)
        samples_for_ticker = 0
        for i in range(len(dates) - 1):
            day = dates[i]
            next_day = dates[i + 1]

            # numeric features: today's close and pct change intraday
            row = hist.loc[hist.index.date == day]
            next_row = hist.loc[hist.index.date == next_day]
            if row.empty or next_row.empty:
                continue

            close = float(row['Close'].iloc[-1])
            open_ = float(row['Open'].iloc[-1])
            change_pct = ((close - open_) / open_) * 100

            # weak label from next-day close
            next_close = float(next_row['Close'].iloc[-1])
            future_change = ((next_close - close) / close) * 100
            if future_change > 0.5:
                label = 2
            elif future_change < -0.5:
                label = 0
            else:
                label = 1

            # fetch news for that ticker around that day (1 day window)
            articles = get_stock_news(ticker, days_back=1, max_articles=3)
            headline = ", ".join([a['title'] or '' for a in articles]) or ""

            texts.append(headline)
            numerics.append([close, change_pct])
            labels.append(label)
            meta.append({"ticker": ticker, "date": day.isoformat(), "text": headline})
            samples_for_ticker += 1
            if samples_for_ticker >= max_per_ticker:
                break

    if not texts:
        raise RuntimeError("No training data collected. Check tickers and availability.")

    X_text = np.array(texts)
    X_num = np.array(numerics, dtype=np.float32)
    y = np.array(labels)
    return X_text, X_num, y, meta


def build_model(vocab_size=20000, embedding_dim=64):
    # Text input pipeline
    text_input = layers.Input(shape=(1,), dtype=tf.string, name='text')
    # Simple TextVectorization
    vectorize = layers.TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=32)
    x = vectorize(text_input)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Numeric input
    num_input = layers.Input(shape=(2,), dtype=tf.float32, name='num')
    n = layers.Dense(16, activation='relu')(num_input)

    # Combine
    combined = layers.concatenate([x, n])
    h = layers.Dense(64, activation='relu')(combined)
    h = layers.Dropout(0.2)(h)
    out = layers.Dense(3, activation='softmax')(h)

    model = models.Model(inputs=[text_input, num_input], outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, vectorize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tickers', nargs='?', help='Comma-separated tickers')
    parser.add_argument('--tickers-file', help='Path to a file with one ticker per line')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-per-ticker', type=int, default=50, help='Limit samples collected per ticker')
    parser.add_argument('--balance', action='store_true', help='Balance classes by undersampling')
    parser.add_argument('--labels-file', help='Path to CSV with human labels. Columns: (ticker,date,label) or (text,label)')
    parser.add_argument('--export-unlabeled', nargs='?', const='unlabeled_samples.csv', help='Export unlabeled samples to CSV for human labeling; optionally provide output path')
    args = parser.parse_args()

    # Determine tickers from positional argument or file
    if args.tickers_file:
        if not os.path.exists(args.tickers_file):
            raise FileNotFoundError(f"Tickers file not found: {args.tickers_file}")
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    else:
        parser.error('Please provide tickers (positional) or --tickers-file')

    print('Collecting data for:', tickers)
    X_text, X_num, y, meta = build_dataset(tickers, max_per_ticker=args.max_per_ticker)
    print('Samples collected (weak labels):', len(y))

    # If requested, export unlabeled samples for human labeling and exit
    if hasattr(args, 'export_unlabeled') and args.export_unlabeled:
        out_path = args.export_unlabeled if isinstance(args.export_unlabeled, str) and args.export_unlabeled else 'unlabeled_samples.csv'
        df = pd.DataFrame(meta)
        df.to_csv(out_path, index=False)
        print(f'Exported {len(df)} unlabeled samples to {out_path}. Please add a `label` column and re-run with --labels-file')
        return

    # If a labels file is provided, load it and replace weak labels with human labels
    if hasattr(args, 'labels_file') and args.labels_file:
        if not os.path.exists(args.labels_file):
            raise FileNotFoundError(f"Labels file not found: {args.labels_file}")
        labels_df = pd.read_csv(args.labels_file)
        # Support two formats: (ticker,date,label) or (text,label)
        labeled_idx = []
        labeled_y = []
        def normalize_label(v):
            # Accept numeric labels 0/1/2 or common strings
            if pd.isna(v):
                return None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    iv = int(v)
                    if iv in (0, 1, 2):
                        return iv
                except Exception:
                    return None
            s = str(v).strip().lower()
            if s in ('2', 'positive', 'pos', 'p', 'bull', 'up'):
                return 2
            if s in ('1', 'neutral', 'neu', 'n', 'flat'):
                return 1
            if s in ('0', 'negative', 'neg', 'negative', 'bear', 'down'):
                return 0
            return None

        for i, m in enumerate(meta):
            matched = None
            # match by ticker+date first
            if {'ticker', 'date', 'label'}.issubset(labels_df.columns):
                rows = labels_df[(labels_df['ticker'].str.upper() == m['ticker'].upper()) & (labels_df['date'] == m['date'])]
                if not rows.empty:
                    val = rows['label'].iloc[0]
                    matched = normalize_label(val)
            # fallback: match by exact text
            if matched is None and {'text', 'label'}.issubset(labels_df.columns):
                rows = labels_df[labels_df['text'] == m['text']]
                if not rows.empty:
                    val = rows['label'].iloc[0]
                    matched = normalize_label(val)
            if matched is not None:
                labeled_idx.append(i)
                labeled_y.append(matched)

        if not labeled_idx:
            raise RuntimeError('No matching labels found in labels file. Make sure columns are (ticker,date,label) or (text,label)')

        # Show a small summary of label counts for verification
        from collections import Counter
        cnts = Counter(labeled_y)
        print('Label counts from human file:', dict(cnts))

        # Filter datasets to only labeled samples and use human labels
        X_text = X_text[labeled_idx]
        X_num = X_num[labeled_idx]
        y = np.array(labeled_y)
        print(f'Using {len(y)} human-labeled samples for training')

    # Optional balancing by undersampling majority classes
    if args.balance:
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        keep_idx = []
        for cls in unique:
            cls_idx = np.where(y == cls)[0]
            if len(cls_idx) > min_count:
                cls_idx = np.random.choice(cls_idx, min_count, replace=False)
            keep_idx.extend(cls_idx.tolist())
        keep_idx = sorted(keep_idx)
        X_text = X_text[keep_idx]
        X_num = X_num[keep_idx]
        y = y[keep_idx]

    if USE_TF:
        model, vectorize = build_model()
        # adapt vectorization
        vectorize.adapt(X_text)

        # Train (small demo)
        history = model.fit({'text': X_text, 'num': X_num}, y, epochs=args.epochs, batch_size=8, validation_split=0.1)

        # Save model
        os.makedirs('models', exist_ok=True)
        model.save('models/sentiment_model')
        print('TensorFlow model saved to models/sentiment_model')
    else:
        # sklearn fallback: tf not available
        print('Using scikit-learn fallback model')
        vectorizer = TfidfVectorizer(max_features=5000)
        scaler = StandardScaler()

        X_text_feat = vectorizer.fit_transform(X_text)
        X_num_scaled = scaler.fit_transform(X_num)
        X_combined = hstack([X_text_feat, X_num_scaled])

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_combined, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump({'vectorizer': vectorizer, 'scaler': scaler, 'clf': clf}, 'models/sentiment_model_sklearn.pkl')
        print('Sklearn model saved to models/sentiment_model_sklearn.pkl')


if __name__ == '__main__':
    main()
