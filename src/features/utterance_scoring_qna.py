import re
from collections import Counter
from dataclasses import asdict
from .qa_pairing import NUMERIC

LM_UNCERTAINTY = {"uncertain","uncertainty","risk","risks","volatility","may","might","could","approximately","around","expect","guidance"}

def score_utterances(utterances, scorer):
    rows = []
    for u in utterances:
        scores, sent = scorer.score_text(u.text, max_tokens=256, stride=64, batch_size=16)
        toks = re.findall(r"\b\w+\b", u.text.lower())
        cnt = Counter(toks); n = max(len(toks), 1)
        unc = sum(cnt[w] for w in LM_UNCERTAINTY)
        numerics = len(NUMERIC.findall(u.text))
        d = asdict(u)
        d.update({
            "neg": scores["negative"], "neu": scores["neutral"], "pos": scores["positive"],
            "polarity": sent, "uncertainty_rate": unc/n, "numeric_density": numerics/n,
            "len_tokens": n
        })
        rows.append(d)
    return rows
