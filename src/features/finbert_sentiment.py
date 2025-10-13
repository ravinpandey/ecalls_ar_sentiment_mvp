from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["negative", "neutral", "positive"]

def _chunk_text(text: str, tokenizer, max_tokens=256, stride=64):
    return tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_tokens,
        stride=stride,
        return_tensors=None,
    )

class FinBertScorer:
    def __init__(self, model_name="ProsusAI/finbert", device: str | None = None, dtype: torch.dtype | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        # fp16 on GPU is fast and safe for inference
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        if self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    @torch.no_grad()
    def score_text(self, text: str, max_tokens=256, stride=64, batch_size=16) -> Tuple[dict, float]:
        if not text or not text.strip():
            zero = {k: 0.0 for k in LABELS}
            return zero, 0.0

        enc = _chunk_text(text, self.tokenizer, max_tokens=max_tokens, stride=stride)
        input_ids_list = enc["input_ids"]
        attn_list = enc["attention_mask"]
        if isinstance(input_ids_list[0], int):
            # single chunk case: wrap into list
            input_ids_list = [input_ids_list]
            attn_list = [attn_list]

        logits_all = []
        # batch over chunks
        for i in range(0, len(input_ids_list), batch_size):
            ids = input_ids_list[i:i+batch_size]
            mask = attn_list[i:i+batch_size]
            batch = self.tokenizer.pad(
                {"input_ids": ids, "attention_mask": mask},
                return_tensors="pt"
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.device == "cuda":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    out = self.model(**batch)
            else:
                out = self.model(**batch)

            logits_all.append(out.logits.detach().to("cpu"))

        logits = torch.cat(logits_all, dim=0)
        probs = torch.softmax(logits, dim=-1).numpy()
        agg = probs.mean(axis=0)

        scores = dict(zip(LABELS, agg.tolist()))
        sent_score = float(agg[2] - agg[0])  # positive - negative
        return scores, sent_score
