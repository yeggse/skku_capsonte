# core/sentimental_classes/sentiment_features.py
import pandas as pd
import numpy as np

def build_sentiment_features(df_news: pd.DataFrame) -> pd.DataFrame:
    df = df_news.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 일자별 mean, count
    daily = df.groupby("date").agg(
        news_count=("sentiment", "count"),
        sentiment_mean=("sentiment", "mean"),
        sentiment_vol=("sentiment", "std"),
    ).fillna(0)

    daily["sentiment_vol"].replace(0, 1e-6, inplace=True)

    return daily.reset_index()
