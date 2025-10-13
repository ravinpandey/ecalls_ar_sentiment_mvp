# scripts/run_finbert_scoring.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.features.finbert_sentiment import FinBertScorer

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run FinBERT sentiment over transcripts parquet.")
    ap.add_argument("--config", default="configs/config.yaml", help="Path to config YAML.")
    ap.add_argument("--in_parquet", default=None, help="Override input parquet path.")
    ap.add_argument("--out_parquet", default=None, help="Override output parquet path.")
    ap.add_argument("--limit", type=int, default=None, help="Score only first N rows (smoke test).")
    ap.add_argument("--batch_size", type=int, default=16, help="Chunk batch size for GPU/CPU scoring.")
    ap.add_argument("--sections", nargs="*", default=None,
                    help="Optional filter: sections to include (e.g., md qa full).")
    return ap.parse_args()

def main():
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    in_path  = args.in_parquet  or f'{cfg["paths"]["interim"]}/transcripts.parquet'
    out_path = args.out_parquet or f'{cfg["paths"]["processed"]}/sentiment_features.parquet'

    # I/O prep
    Path(cfg["paths"]["processed"]).mkdir(parents=True, exist_ok=True)

    # Load transcripts
    df = pd.read_parquet(in_path)

    # Optional filters
    if args.sections:
        keep = set(s.lower() for s in args.sections)
        df = df[df["section"].str.lower().isin(keep)]
    if args.limit:
        df = df.head(args.limit).copy()

    if df.empty:
        print("[finbert] Nothing to score (empty input after filters).")
        return

    # Init scorer (auto-detects CUDA and uses fp16 if available)
    scorer = FinBertScorer(
        model_name=cfg["finbert"]["model_name"]
    )

    # Process rows
    records = []
    # (tqdm is optional; avoid import if not installed)
    try:
        from tqdm import tqdm
        iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Scoring")
    except Exception:
        iterator = df.itertuples(index=False)

    for r in iterator:
        scores, sent = scorer.score_text(
            r.text,
            max_tokens=cfg["finbert"]["max_tokens"],
            stride=cfg["finbert"]["stride"],
            batch_size=args.batch_size,
        )
        records.append({
            "ticker": r.ticker,
            "call_date": str(r.call_date) if pd.notna(r.call_date) else None,
            "section": r.section,
            "sent_score": sent,  # positive - negative
            "negative": scores.get("negative", 0.0),
            "neutral":  scores.get("neutral",  0.0),
            "positive": scores.get("positive", 0.0),
        })

    feat = pd.DataFrame.from_records(records)

    # Write output
    engine = None
    try:
        import pyarrow  # noqa: F401
        engine = "pyarrow"
    except Exception:
        pass

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out_path, index=False, engine=engine)
    print(f"[finbert] wrote: {out_path} | rows: {len(feat)} | cols: {list(feat.columns)}")

if __name__ == "__main__":
    main()
