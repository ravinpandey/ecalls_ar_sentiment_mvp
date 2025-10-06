# 🧠 Earnings Call Sentiment vs Abnormal Returns (MVP)

This project investigates the relationship between **Abnormal Returns (AR)** and **Sentiment** extracted from **Earnings Call Transcripts**.

The goal of this Minimal Viable Product (MVP) is to **validate whether the tone or sentiment of an earnings call correlates with post-call abnormal returns** over short windows *(1, 3, 7 days)*.

---

## 🎯 Research Objective

To test the hypothesis:

> “Positive sentiment in earnings call transcripts is associated with positive abnormal stock returns (AR) in the short-term window following the call.”

---

## 🧩 Methodology

1. **Collect Data**  
   - Download daily closing prices using `yfinance` for each stock and benchmark (e.g., `SPY`).  
   - Prepare transcripts in CSV with columns: `ticker, call_date, text`.

2. **Compute Sentiment**  
   - Use **FinBERT** (`ProsusAI/finbert`) to compute `positive`, `negative`, `neutral`, and `compound` sentiment scores.

3. **Compute Abnormal Returns (AR)**  
   \[
   R = \frac{P_{t+k} - P_t}{P_t}, \quad
   AR = R_{\text{stock}} - R_{\text{benchmark}}, \quad k \in \{1, 3, 7\}
   \]

4. **Analyze Relationship**  
   - Correlate `compound` sentiment score with AR values.  
   - Perform OLS regression and visualize results with scatter plots.

---

## 🚀 Quickstart

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
