import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.agents_set import dir_info


model_dir: str = dir_info["model_dir"]
data_dir: str = dir_info["data_dir"]
OUTPUT_DIR = data_dir


class MacroAData:
    """거시경제(Macro) 데이터셋 생성 및 전처리 클래스"""

    def __init__(self, ticker="NVDA"):
        self.merged_df = None
        self.macro_tickers = {
            "SPY": "SPY", "QQQ": "QQQ", "^GSPC": "^GSPC", "^DJI": "^DJI", "^IXIC": "^IXIC",
            "DX-Y.NYB": "DX-Y.NYB", "EURUSD=X": "EURUSD=X", "USDJPY=X": "USDJPY=X",
            "^TNX": "^TNX", "^IRX": "^IRX", "^FVX": "^FVX", "^VIX": "^VIX",
            "GC=F": "GC=F", "CL=F": "CL=F", "HG=F": "HG=F",
        }
        self.data = None
        self.agent_id = "MacroAgent"
        self.ticker = ticker
        self.model_path = f"{model_dir}/{self.ticker}_{self.agent_id}.pt"
        self.scaler_X_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_xscaler.pkl"
        self.scaler_y_path = f"{model_dir}/scalers/{self.ticker}_{self.agent_id}_yscaler.pkl"

        five_years_ago = datetime.today() - relativedelta(years=5)
        self.start_date = five_years_ago.strftime("%Y-%m-%d")
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def fetch_data(self):
        """거시경제 데이터 다운로드 및 컬럼 평탄화"""
        df = yf.download(
            tickers=list(self.macro_tickers.values()),
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df = df.stack(level=0)
            df.index.names = ["Date", "Ticker"]
            df = df.unstack(level="Ticker")
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns.values]
        else:
            df.index.name = "Date"

        self.data = df
        print(f"[MacroAgent] Data shape: {df.shape}, Columns: {len(df.columns)}")
        return df

    def add_features(self):
        """수익률, 금리차, 위험심리 및 주식 특성 추가"""
        df = self.data.copy()

        for ticker in self.macro_tickers.values():
            col_close = f"{ticker}_Close"
            col_ret = f"{ticker}_ret_1d"
            if col_close in df.columns:
                df[col_ret] = df[col_close].pct_change()

        if "^TNX_Close" in df.columns and "^IRX_Close" in df.columns:
            df["Yield_spread"] = df["^TNX_Close"] - df["^IRX_Close"]

        if {"SPY_ret_1d", "DX-Y.NYB_ret_1d", "^VIX_ret_1d"} <= set(df.columns):
            df["Risk_Sentiment"] = (
                    df["SPY_ret_1d"] - df["DX-Y.NYB_ret_1d"] - df["^VIX_ret_1d"]
            )

        df_stock_price = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            progress=False
        )[["Close"]].copy()

        if isinstance(df_stock_price.columns, pd.MultiIndex):
            df_stock_price.columns = ["Close"]
        df_stock_price.index.name = "Date"
        df_stock_price = df_stock_price.reset_index()
        df_stock_price["Date"] = pd.to_datetime(df_stock_price["Date"]).dt.strftime("%Y-%m-%d")

        t = self.ticker
        df_stock_price["ret1"] = df_stock_price["Close"].pct_change()
        df_stock_price["ma5"] = df_stock_price["Close"].rolling(5).mean()
        df_stock_price["ma10"] = df_stock_price["Close"].rolling(10).mean()
        df_stock_price = df_stock_price.rename(columns={"Close": t})

        df_stock_price_features = df_stock_price[["Date", t, "ret1", "ma5", "ma10"]]

        df_macro_raw = df.copy().reset_index()
        if isinstance(df_macro_raw.columns, pd.MultiIndex):
            df_macro_raw.columns = df_macro_raw.columns.get_level_values(-1)

        df_macro_raw["Date"] = pd.to_datetime(df_macro_raw["Date"]).dt.strftime("%Y-%m-%d")
        merged = pd.merge(df_macro_raw, df_stock_price_features, on="Date", how="inner")
        merged = merged.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)

        self.data = merged
        print(f"[INFO] Feature engineering complete: {self.data.shape}")
        return self.data

    def save_csv(self):
        """전처리된 데이터를 raw 폴더에 저장"""
        raw_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        path = os.path.join(raw_dir, f"{self.ticker}_{self.agent_id}_raw.csv")

        df = self.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)) for col in df.columns]
        df.columns = [str(c).strip() for c in df.columns]

        if "Date" not in df.columns:
            df["Date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")

        cols = ["Date"] + [c for c in df.columns if c != "Date"]
        df = df[cols]

        t = self.ticker
        if t in df.columns:
            df["Close"] = df[t]
        cols = [c for c in df.columns if c != "Close"] + ["Close"]
        df = df[cols]

        df.to_csv(path, index=False)
        print(f"[MacroAgent] Saved: {path}")

    def make_close_price(self):
        """일별 종가 저장"""
        df_prices = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False
        )[["Close"]].reset_index()

        df_prices["Date"] = pd.to_datetime(df_prices["Date"]).dt.strftime("%Y-%m-%d")
        path = os.path.join(OUTPUT_DIR, "daily_closePrice.csv")
        df_prices.to_csv(path, index=False)

        print("[make_close_price] 저장 완료:", df_prices.shape, "rows at", path)

    def model_maker(self):
        """MacroAData: 모델 학습용 데이터셋 생성 및 스케일러 저장"""

        # -------------------------------------------------------------
        # 1. 데이터 로드
        # -------------------------------------------------------------
        raw_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "raw")
        macro_path = os.path.join(raw_dir, f"{self.ticker}_{self.agent_id}_raw.csv")
        price_path = os.path.join(OUTPUT_DIR, "daily_closePrice.csv")

        macro_df = pd.read_csv(macro_path)
        price_df = pd.read_csv(price_path)

        # 날짜 컬럼 정리
        macro_df["Date"] = pd.to_datetime(macro_df["Date"]).dt.strftime("%Y-%m-%d")
        if "Date" not in price_df.columns:
            price_df = price_df.reset_index().rename(columns={"index": "Date"})
        price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.strftime("%Y-%m-%d")

        # 매크로 데이터의 'Close' → 종목명으로 변경
        if "Close" in macro_df.columns:
            macro_df = macro_df.rename(columns={"Close": self.ticker})

        # 병합 (Date 기준)
        merged = pd.merge(price_df, macro_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

        # -------------------------------------------------------------
        # 2. 디버깅용 컬럼 확인
        # -------------------------------------------------------------
        print(f"[DEBUG] merged columns sample: {list(merged.columns)[:15]}")

        # -------------------------------------------------------------
        # 3. 피처 컬럼 선택
        # -------------------------------------------------------------
        all_numeric_cols = merged.select_dtypes(include=["number"]).columns.tolist()
        cols_to_exclude = {self.ticker, f"{self.ticker}_target"}
        feature_cols = [c for c in all_numeric_cols if c not in cols_to_exclude]

        X_all = merged[feature_cols].fillna(0)

        # -------------------------------------------------------------
        # 4. Volume 및 상수 컬럼 제거
        # -------------------------------------------------------------
        remove_patterns = ["Volume_", "Unnamed:"]
        X_all = X_all.drop(
            columns=[c for c in X_all.columns if any(p in c for p in remove_patterns)],
            errors="ignore"
        )

        constant_cols = []
        for c in X_all.columns:
            std_val = X_all[c].std()
            if isinstance(std_val, pd.Series):
                std_val = std_val.mean()
            if np.isclose(std_val, 0.0):
                constant_cols.append(c)

        if constant_cols:
            X_all = X_all.drop(columns=constant_cols, errors="ignore")

        feature_cols = X_all.columns.tolist()

        # -------------------------------------------------------------
        # 5. 입력 스케일링
        # -------------------------------------------------------------
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # -------------------------------------------------------------
        # 6. 타깃(Target) 생성
        # -------------------------------------------------------------
        possible_cols = [self.ticker, f"{self.ticker}_Close", "Close"]
        price_series = None

        for col in possible_cols:
            if col in merged.columns:
                price_series = merged[col]
                # 혹시 DataFrame일 경우 첫 열만 선택
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series.iloc[:, 0]
                break

        if price_series is None:
            raise KeyError(f"종가 관련 컬럼을 찾을 수 없습니다. available={list(merged.columns)[:15]}")

        # 다음날 수익률 생성
        merged[f"{self.ticker}_target"] = price_series.astype(float).pct_change().shift(-1)

        # -------------------------------------------------------------
        # 7. y 스케일링
        # -------------------------------------------------------------
        y_all = merged[[f"{self.ticker}_target"]].dropna().reset_index(drop=True)
        X_scaled = X_scaled.iloc[:len(y_all)]

        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = scaler_y.fit_transform(y_all)

        # -------------------------------------------------------------
        # 8. 시퀀스 데이터 생성
        # -------------------------------------------------------------
        def create_sequences(X, y, window=40):
            Xs, ys = [], []
            for i in range(len(X) - window):
                Xs.append(X.iloc[i:(i + window)].values)
                ys.append(y[i + window])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, window=40)

        # -------------------------------------------------------------
        # 9. Train/Test 분리
        # -------------------------------------------------------------
        split_idx = int(len(X_seq) * 0.8)
        self.X_train, self.X_test = X_seq[:split_idx], X_seq[split_idx:]
        self.y_train, self.y_test = y_seq[:split_idx], y_seq[split_idx:]

        # -------------------------------------------------------------
        # 10. 스케일러 저장
        # -------------------------------------------------------------
        os.makedirs(os.path.dirname(self.scaler_X_path), exist_ok=True)
        scaler_X.feature_names_in_ = np.array(feature_cols)

        joblib.dump(scaler_X, self.scaler_X_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        # -------------------------------------------------------------
        # 11. 완료 로그
        # -------------------------------------------------------------
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        print(f"[OK] MacroAData.model_maker 완료: {len(X_seq)} samples, {len(feature_cols)} features")
