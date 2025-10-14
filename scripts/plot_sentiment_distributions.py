# save as scripts/plot_sentiment_distributions.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

feat_path = "data/processed/sentiment_features.parquet"
out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_parquet(feat_path)

# Quick stats
stats = df[["sent_score", "negative", "neutral", "positive"]].describe().T
print("\nSummary stats:\n", stats)

def save_hist(series, title, xlabel, outfile, bins=30):
    plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()

# Make plots
overall_path = out_dir / "sentiment_dist_overall.png"
pos_path     = out_dir / "sentiment_dist_positive.png"
neg_path     = out_dir / "sentiment_dist_negative.png"
neu_path     = out_dir / "sentiment_dist_neutral.png"

save_hist(df["sent_score"], "Distribution of Overall Sentiment Score (positive - negative)", "sent_score", overall_path)
save_hist(df["positive"],   "Distribution of Positive Probability", "positive", pos_path)
save_hist(df["negative"],   "Distribution of Negative Probability", "negative", neg_path)
save_hist(df["neutral"],    "Distribution of Neutral Probability", "neutral", neu_path)

# Bundle to a single PDF (one per page)
pdf_path = out_dir / "sentiment_distributions.pdf"
with PdfPages(pdf_path) as pdf:
    for p in [overall_path, pos_path, neg_path, neu_path]:
        img = plt.imread(p)
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig()
        plt.close()

print("\nSaved:")
print(f"- {overall_path}")
print(f"- {pos_path}")
print(f"- {neg_path}")
print(f"- {neu_path}")
print(f"- {pdf_path}")
