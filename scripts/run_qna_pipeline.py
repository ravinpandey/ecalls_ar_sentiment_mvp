import os, glob, pandas as pd
from src.transcript.parser import parse_transcript
from src.sentiment.finbert_scorer import FinBertScorer
from src.features.utterance_scoring_qna import score_utterances
from src.features.qa_pairing import build_qa_pairs

RAW = "data/raw/Earnings Call Transcripts – NASDAQ (2016-2020)/Transcripts"

def read_txt(fp): 
    return open(fp, 'r', encoding='utf-8', errors='ignore').read()

def guess_call_id(fp):
    # e.g., data/.../AAPL/2017-Aug-01-AAPL.txt -> AAPL_2017-Aug-01
    base = os.path.basename(fp).replace(".txt","")
    parts = base.split("-")
    ticker = parts[-1]
    date = "-".join(parts[:3])
    return f"{ticker}_{date}"

def main():
    print("Parsing → scoring → pairing ...")
    rows = []
    for fp in glob.glob(os.path.join(RAW, "*", "*.txt")):
        if fp.endswith(":Zone.Identifier"): 
            continue
        call_id = guess_call_id(fp)
        raw = read_txt(fp)
        utts = parse_transcript(call_id, raw)
        rows.extend([u.__dict__ for u in utts])

    df_u = pd.DataFrame(rows)
    os.makedirs("data/interim", exist_ok=True)
    df_u.to_parquet("data/interim/utterances.parquet", index=False)

    scorer = FinBertScorer()
    # quick dataclass adapter not needed; we already produced dicts
    from types import SimpleNamespace
    utt_objs = [SimpleNamespace(**r) for r in rows]
    scored = score_utterances(utt_objs, scorer)
    df_s = pd.DataFrame(scored)
    os.makedirs("data/processed", exist_ok=True)
    df_s.to_parquet("data/processed/utterances_scored.parquet", index=False)

    pairs = build_qa_pairs(df_s.to_dict(orient="records"))
    pd.DataFrame(pairs).to_parquet("data/processed/qa_pairs.parquet", index=False)

    print("Wrote:")
    print("  data/interim/utterances.parquet")
    print("  data/processed/utterances_scored.parquet")
    print("  data/processed/qa_pairs.parquet")

if __name__ == "__main__":
    main()
