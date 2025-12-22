# core/sentimental_classes/finbert_scorer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

class FinBertScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def score(self, df_news: pd.DataFrame) -> pd.DataFrame:
        scores = []
        for text in df_news["content"].fillna(""):
            inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].numpy()
            # positive - negative 로 score
            score = probs[0] - probs[2]
            scores.append(score)

        df_news["sentiment"] = scores
        return df_news
