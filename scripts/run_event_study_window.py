from datetime import timedelta, date
from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml
from tqdm import tqdm

# ---------- helper functions ----------

def get_returns(ticker, start, end):
    """daily returns between start and end"""
    px = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    return px.pct_change().dropna()

def abnormal_returns(stock_ret, mkt_ret):
    df = pd.concat([stock_ret, mkt_ret], axis=1).dropna()
    df.columns = ["stock", "mkt"]
    df["AR"] = df["stock"] - df["mkt"]
    return df

def car_after_event(ticker, event_date, market="^GSPC", days=[1,3,7]):
    """returns dict with CAR+1,+3,+7 after event_date"""
    event_date = pd.to_datetime(event_date).date()
    start = event_date - timedelta(days=15)
    end   = event_date + timedelta(days=max(days)+10)
    sret = get_returns(ticker, start, end)
    mret = get_returns(market, start, end)
    ar = abnormal_returns(sret, mret)
    out = {}
    for d in days:
        mask = (ar.index.date > event_date) & (ar.index.date <= event_date + timedelta(days=d))
        out[f"CAR+{d}"] = ar.loc[mask, "AR"].sum()
    return out

# ---------- main ----------
if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    feat_path = Path(cfg["paths"]["processed"]) / "sentiment_features.parquet"
    df = pd.read_parquet(feat_path)

    df = df.dropna(subset=["call_date"]).copy()
    results = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Event Study"):
        try:
            car_vals = car_after_event(r.ticker, r.call_date, market=cfg["event_study"]["market"])
            results.append({
                "ticker": r.ticker,
                "call_date": str(r.call_date),
                "sent_score": r.sent_score,
                **car_vals
            })
        except Exception as e:
            print("skip", r.ticker, r.call_date, "->", e)

    out = pd.DataFrame(results)
    out_path = Path(cfg["paths"]["reports"]) / "event_study_CARs.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[event-study] wrote {out_path}  | rows: {len(out)}")

    # correlation view
    print("\nCorrelation of sentiment with CARs:")
    print(out[["sent_score","CAR+1","CAR+3","CAR+7"]].corr().round(3))
