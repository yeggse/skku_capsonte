import os
import numpy as np
import pandas as pd
import yfinance as yf

from config.agents_set import agents_info, common_params


MACRO_TICKERS = {
    "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
    "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX",
    "^VIX": "^VIX",
    "DX-Y.NYB": "DX-Y.NYB",
    "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
    "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
}


class MacroCSVBuilder:
    """
    MacroAgent 전용 Raw CSV 생성기
    - 매크로 지표 + 개별 종목 병합
    - feature 계산
    - Raw CSV 저장
    """

    def __init__(self, agent_id: str, data_dir: str):
        self.agent_id = agent_id
        self.data_dir = data_dir

    def ensure_csv(self, ticker: str, rebuild: bool = False) -> str:
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")

        if not rebuild and os.path.exists(csv_path):
            return csv_path

        print(f"[{self.agent_id}] Raw CSV 생성 중...")

        period = common_params.get("period", "2y")

        # -------------------------------------------------
        # 1. 매크로 데이터 다운로드
        # -------------------------------------------------
        df_macro = yf.download(
            tickers=list(MACRO_TICKERS.values()),
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )

        if df_macro.empty:
            raise ValueError("매크로 데이터 다운로드 실패")

        # MultiIndex → 단일 컬럼
        if isinstance(df_macro.columns, pd.MultiIndex):
            df_macro = df_macro.stack(level=0)
            df_macro.index.names = ["Date", "Ticker"]
            df_macro = df_macro.unstack(level="Ticker")
            df_macro.columns = [f"{c[1]}_{c[0]}" for c in df_macro.columns.values]

        df_macro = df_macro.reset_index()
        df_macro["Date"] = pd.to_datetime(df_macro["Date"])

        # -------------------------------------------------
        # 2. 매크로 파생 피처
        # -------------------------------------------------
        df_macro_feat = df_macro.set_index("Date")

        for t in MACRO_TICKERS.values():
            close_col = f"{t}_Close"
            if close_col in df_macro_feat.columns:
                df_macro_feat[f"{t}_ret_1d"] = df_macro_feat[close_col].pct_change()

        if "^TNX_Close" in df_macro_feat.columns and "^IRX_Close" in df_macro_feat.columns:
            df_macro_feat["Yield_spread"] = (
                    df_macro_feat["^TNX_Close"] - df_macro_feat["^IRX_Close"]
            )

        if (
                "SPY_ret_1d" in df_macro_feat.columns
                and "DX-Y.NYB_ret_1d" in df_macro_feat.columns
                and "^VIX_ret_1d" in df_macro_feat.columns
        ):
            df_macro_feat["Risk_Sentiment"] = (
                    df_macro_feat["SPY_ret_1d"]
                    - df_macro_feat["DX-Y.NYB_ret_1d"]
                    - df_macro_feat["^VIX_ret_1d"]
            )

        df_macro_feat = df_macro_feat.reset_index()

        # -------------------------------------------------
        # 3. 개별 종목 가격 데이터
        # -------------------------------------------------
        df_price = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df_price.empty:
            raise ValueError(f"{ticker} 가격 데이터 없음")

        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df_price.columns]

        df_price = df_price[["Close"]]
        df_price.index.name = "Date"
        df_price = df_price.reset_index()
        df_price["Date"] = pd.to_datetime(df_price["Date"])
        df_price = df_price.rename(columns={"Close": ticker})

        df_price["ret1"] = df_price[ticker].pct_change()
        df_price["ma5"] = df_price[ticker].rolling(5).mean()
        df_price["ma10"] = df_price[ticker].rolling(10).mean()
        df_price = df_price.fillna(method="bfill")

        # -------------------------------------------------
        # 4. 병합
        # -------------------------------------------------
        merged = pd.merge(df_price, df_macro_feat, on="Date", how="inner")
        merged = merged.sort_values("Date")
        merged = merged.ffill().bfill().dropna().reset_index(drop=True)

        # -------------------------------------------------
        # 5. feature 컬럼 정렬
        # -------------------------------------------------
        cfg = agents_info.get(self.agent_id, {})
        feature_cols = cfg.get("data_cols", [])
        if not feature_cols:
            raise ValueError(f"[{self.agent_id}] data_cols not defined in config")

        X_df = pd.DataFrame(index=merged.index)
        for col in feature_cols:
            X_df[col] = merged[col] if col in merged.columns else 0.0

        merged["Close"] = merged[ticker] if ticker in merged.columns else np.nan

        out_df = pd.concat(
            [
                merged[["Date"]].reset_index(drop=True),
                X_df.reset_index(drop=True),
                merged[["Close"]].reset_index(drop=True),
            ],
            axis=1,
        )

        # -------------------------------------------------
        # 6. 기간 필터링
        # -------------------------------------------------
        end_date = pd.Timestamp.today().normalize()
        days = self._resolve_period_days(period)
        start_date = end_date - pd.Timedelta(days=days)

        out_df = out_df[out_df["Date"] >= start_date]
        out_df = out_df.sort_values("Date").reset_index(drop=True)
        out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")

        out_df.to_csv(csv_path, index=False)

        print(f"✅ [{self.agent_id}] Raw CSV 저장 완료: {csv_path} ({len(out_df):,} rows)")
        return csv_path

    @staticmethod
    def _resolve_period_days(period: str) -> int:
        if period.endswith("y"):
            return int(period[:-1]) * 365
        if period.endswith("m"):
            return int(period[:-1]) * 30
        if period.endswith("d"):
            return int(period[:-1])
        return 2 * 365
