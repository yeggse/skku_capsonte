import os
import pandas as pd
import yfinance as yf

from config.agents_set import common_params, agents_info
from core.sentimental_classes.news import update_news_db
from core.sentimental_classes.sentimental_feature_builder import SentimentalFeatureBuilder


class SentimentalCSVBuilder:
    """
    SentimentalAgent 전용 Raw CSV 생성기
    """

    def __init__(self, agent_id: str, data_dir: str, news_dir: str):
        self.agent_id = agent_id
        self.data_dir = data_dir
        self.news_dir = news_dir

    def ensure_csv(self, ticker: str, rebuild: bool = False) -> str:
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")

        if not rebuild and os.path.exists(csv_path):
            return csv_path

        print(f"[{self.agent_id}] Raw CSV 생성 중...")

        days = self._resolve_period_days()
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # -----------------------
        # 가격 데이터
        # -----------------------
        df_price = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if df_price.empty:
            raise ValueError("가격 데이터 없음")

        # yfinance MultiIndex 대응
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [
                col[0].lower() if isinstance(col, tuple) else str(col).lower()
                for col in df_price.columns
            ]
        else:
            df_price.columns = [str(c).lower() for c in df_price.columns]

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # -----------------------
        # 뉴스 데이터
        # -----------------------
        price_start_date = pd.to_datetime(df_price["date"]).min()
        df_news = update_news_db(
            ticker=ticker,
            base_dir=self.news_dir,
            target_start_date=price_start_date
        )

        # -----------------------
        # Feature 생성
        # -----------------------
        df_feat = SentimentalFeatureBuilder.build(df_price, df_news)

        # -----------------------
        # CSV 저장
        # -----------------------
        cfg = agents_info[self.agent_id]
        feature_cols = cfg.get("data_cols", [])

        df_out = df_feat.copy()
        df_out["Date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d")
        df_out["Close"] = df_out["close"]

        cols = ["Date"] + feature_cols + ["Close"]
        df_out[cols].to_csv(csv_path, index=False)

        print(f"✅ [{self.agent_id}] Raw CSV 저장 완료: {csv_path} ({len(df_out):,} rows)")
        return csv_path

    def _resolve_period_days(self) -> int:
        period = common_params.get("period", "2y")
        if period.endswith("y"):
            return int(period[:-1]) * 365
        if period.endswith("m"):
            return int(period[:-1]) * 30
        if period.endswith("d"):
            return int(period[:-1])
        return 2 * 365
