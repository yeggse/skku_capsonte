import numpy as np
import pandas as pd


class SentimentalFeatureBuilder:
    """
    뉴스 + 가격 데이터를 받아 SentimentalAgent용 Feature DataFrame 생성
    """

    @staticmethod
    def build(df_price: pd.DataFrame, df_news: pd.DataFrame) -> pd.DataFrame:
        df = df_price.copy()

        # -----------------------
        # 뉴스 피처 병합
        # -----------------------
        if df_news is None or df_news.empty:
            for col in [
                "news_count_1d", "sentiment_mean_1d",
                "news_count_7d", "sentiment_mean_7d", "sentiment_vol_7d"
            ]:
                df[col] = 0.0
        else:
            daily = df_news.groupby("date").agg(
                count=("sentiment_score", "count"),
                mean=("sentiment_score", "mean")
            )

            idx = pd.date_range(daily.index.min(), daily.index.max())
            daily = daily.reindex(idx, fill_value=0)
            daily.index.name = "date"

            daily["news_count_1d"] = daily["count"]
            daily["sentiment_mean_1d"] = daily["mean"]
            daily["news_count_7d"] = daily["count"].rolling(7).sum().fillna(0)
            daily["sentiment_mean_7d"] = daily["mean"].rolling(7).mean().fillna(0)
            daily["sentiment_vol_7d"] = daily["mean"].rolling(7).std().fillna(0)

            daily = daily.reset_index()

            df = df.merge(daily, on="date", how="left")

            for col in [
                "news_count_1d", "sentiment_mean_1d",
                "news_count_7d", "sentiment_mean_7d", "sentiment_vol_7d"
            ]:
                df[col] = df[col].fillna(0.0)

        # -----------------------
        # 가격 기반 피처
        # -----------------------
        df["return_1d"] = df["close"].pct_change().fillna(0)
        df["hl_range"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).fillna(0)
        df["Volume"] = df["volume"].fillna(0)

        return df.sort_values("date").reset_index(drop=True)
