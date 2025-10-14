import re
NUMERIC = re.compile(r"(\$?\d[\d,]*(\.\d+)?%?)")

def build_qa_pairs(scored_rows):
    rows = sorted(scored_rows, key=lambda x: (x["call_id"], x["order"]))
    qa_pairs, current_q = [], None
    current_call = None
    for r in rows:
        if r["section"]!="qa": 
            continue
        if current_call != r["call_id"]:
            current_call = r["call_id"]; current_q = None
        if r["turn_type"]=="question":
            if current_q: qa_pairs.append(current_q)
            current_q = {"q": r, "answers": []}
        elif r["turn_type"]=="answer" and current_q:
            current_q["answers"].append(r)
    if current_q: qa_pairs.append(current_q)

    pairs = []
    for p in qa_pairs:
        if not p["answers"]: continue
        A = p["answers"][0]  # simplest policy; you can change to aggregate
        Q = p["q"]
        pairs.append({
            "call_id": Q["call_id"],
            "q_order": Q["order"],
            "q_polarity": Q["polarity"], "a_polarity": A["polarity"],
            "qa_delta": A["polarity"] - Q["polarity"],
            "q_len": Q["len_tokens"], "a_len": A["len_tokens"],
            "len_ratio": (A["len_tokens"]+1e-6)/(Q["len_tokens"]+1e-6),
            "q_unc": Q["uncertainty_rate"], "a_unc": A["uncertainty_rate"],
            "unc_delta": A["uncertainty_rate"] - Q["uncertainty_rate"],
            "q_num": Q["numeric_density"], "a_num": A["numeric_density"],
            "num_delta": A["numeric_density"] - Q["numeric_density"],
            "answer_speaker_role": A["speaker_role"],
            "q_text": Q["text"], "a_text": A["text"]
        })
    return pairs
