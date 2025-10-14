from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]

def _chunk(text, tok, max_tokens=256, stride=64):
    return tok(text, return_overflowing_tokens=True, truncation=True,
               max_length=max_tokens, stride=stride, return_tensors=None)

class FinBertScorer:
    def __init__(self, model_name="ProsusAI/finbert", device: str|None=None, dtype: torch.dtype|None=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.dtype = dtype or (torch.float16 if self.device=="cuda" else torch.float32)
        if self.device=="cuda":
            try: torch.set_float32_matmul_precision("high")
            except Exception: pass

    @torch.no_grad()
    def score_text(self, text: str, max_tokens=256, stride=64, batch_size=16) -> Tuple[dict, float]:
        if not text or not text.strip():
            zero = {k: 0.0 for k in LABELS}; return zero, 0.0
        enc = _chunk(text, self.tok, max_tokens, stride)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        if isinstance(ids[0], int): ids, attn = [ids], [attn]
        logits_all = []
        for i in range(0, len(ids), batch_size):
            batch = self.tok.pad({"input_ids": ids[i:i+batch_size], "attention_mask": attn[i:i+batch_size]}, return_tensors="pt")
            batch = {k:v.to(self.device) for k,v in batch.items()}
            if self.device=="cuda":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    out = self.model(**batch)
            else:
                out = self.model(**batch)
            logits_all.append(out.logits.detach().cpu())
        import torch as T
        logits = T.cat(logits_all, dim=0)
        probs = T.softmax(logits, dim=-1).numpy().mean(0)
        scores = dict(zip(LABELS, probs.tolist()))
        return scores, float(probs[2]-probs[0])
