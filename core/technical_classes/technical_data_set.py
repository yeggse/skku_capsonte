# core/technical_classes/technical_data_set.py

import pandas as pd
import numpy as np
import os
import warnings
import yfinance as yf
from config.agents_set import agents_info, dir_info

import importlib # 아연수정

# yfinance 진행률 바 및 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# Raw Dataset 생성
def fetch_ticker_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame: # 아연수정
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df.dropna(inplace=True)

    # 컬럼명 정리 (튜플 형태를 단순 문자열로 변환)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 기본 기술적 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

    # Fundamental Agent용 추가 데이터
    try:
        # 환율 데이터 (USD/KRW)
        usd_krw = yf.download("USDKRW=X", period=period, interval=interval, auto_adjust=False, progress=False)
        if not usd_krw.empty:
            df["USD_KRW"] = usd_krw["Close"].reindex(df.index, method='ffill')
        else:
            df["USD_KRW"] = 1300.0  # 기본값

        # 나스닥 지수
        nasdaq = yf.download("^IXIC", period=period, interval=interval, auto_adjust=False, progress=False)
        if not nasdaq.empty:
            df["NASDAQ"] = nasdaq["Close"].reindex(df.index, method='ffill')
        else:
            df["NASDAQ"] = 15000.0  # 기본값

        # VIX 지수
        vix = yf.download("^VIX", period=period, interval=interval, auto_adjust=False, progress=False)
        if not vix.empty:
            df["VIX"] = vix["Close"].reindex(df.index, method='ffill')
        else:
            df["VIX"] = 20.0  # 기본값
    except Exception as e:
        print(f"⚠️ 추가 지표 다운로드 실패: {e}")
        df["USD_KRW"] = 1300.0
        df["NASDAQ"] = 15000.0
        df["VIX"] = 20.0



    # Sentimental Agent용 감성 지표
    df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
    df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)

    df.dropna(inplace=True)
    return df

# 시퀀스 생성
def create_sequences(features, target, window_size=14):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

#내부: feature builder 함수 로더 (테크니컬 전용 포함) (아연수정)
def _load_builder(spec: str):
    # "core.features.technical:build_features_technical"
    mod, func = spec.split(":")
    return getattr(importlib.import_module(mod), func)

# 통합 함수 (아연수정)
def build_dataset(ticker: str = "TSLA", save_dir=dir_info["data_dir"], period: str = "5y", interval: str = "1d"):
    os.makedirs(save_dir, exist_ok=True)
    # raw 테크니컬 피처 저장용 디렉토리 (data/raw)
    raw_root = os.path.join(os.path.dirname(save_dir), "raw")
    os.makedirs(raw_root, exist_ok=True)
    df = fetch_ticker_data(ticker, period=period, interval=interval)
    # 원시 OHLCV + 공통 파생지표는 여전히 processed/raw_data 경로에 저장
    df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)

    for agent_id, cfg in agents_info.items():
        df_agent = df.copy()

        # TechnicalAgent에만 전용 feature_builder 적용
        spec = cfg.get("feature_builder") if agent_id == "TechnicalAgent" else None
        if spec:
            builder = _load_builder(spec)
            feat = builder(df_agent[["Open", "High", "Low", "Close", "Volume"]])
            if not isinstance(feat, pd.DataFrame):
                raise TypeError("feature_builder는 DataFrame을 반환해야 합니다.")

            # === (1) 스케일링/윈도우 처리 전 RAW 테크니컬 피처를 data/raw에 저장 ===
            # 형식 통일:
            #   - 첫 컬럼: Date
            #   - 마지막 컬럼: 종가(Close)
            try:
                raw_tech = feat.copy()

                # 인덱스를 Date 컬럼으로 복원
                raw_tech.index.name = "Date"
                raw_tech.reset_index(inplace=True)

                # 필요 시 ticker 컬럼 유지
                if "ticker" not in raw_tech.columns:
                    raw_tech.insert(1, "ticker", ticker)

                # 원본 df_agent 에서 Close 를 가져와 마지막 컬럼으로 추가
                try:
                    close_df = df_agent[["Close"]].copy()
                    close_df.index.name = "Date"
                    close_df.reset_index(inplace=True)
                    # Date 기준으로 병합 (이미 정렬된 상태)
                    raw_tech = raw_tech.merge(close_df, on="Date", how="left")
                except Exception as e:
                    print(f"⚠️ Failed to attach Close column for TechnicalAgent raw features: {e}")

                # Close 를 항상 마지막 컬럼으로 이동
                if "Close" in raw_tech.columns:
                    cols = [c for c in raw_tech.columns if c != "Close"] + ["Close"]
                    raw_tech = raw_tech[cols]

                raw_csv_path = os.path.join(raw_root, f"{ticker}_TechnicalAgent_raw.csv")
                raw_tech.to_csv(raw_csv_path, index=False)
                print(f"✅ {ticker} TechnicalAgent raw features saved to {raw_csv_path} ({len(raw_tech)} rows)")
            except Exception as e:
                print(f"⚠️ Failed to save TechnicalAgent raw features: {e}")

            # === (2) 기존 로직: 테크니컬 피처를 원본 df에 조인 후 processed용 시퀀스 생성 ===
            df_agent = (
                df_agent.join(feat, how="left")
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .dropna()
            )

        col = cfg["data_cols"]
        X = df_agent[col]

        returns = df_agent["Close"].pct_change().shift(-1)
        valid_mask = ~returns.isna()
        y = returns[valid_mask].values.reshape(-1, 1)
        X = X[valid_mask]

        X_seq, y_seq = create_sequences(X, y, window_size=cfg["window_size"])
        samples, time_steps, features = X_seq.shape
        print(f"[{agent_id}] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

        # 평탄화(csv 저장용)
        flattened_data = []

        # 윈도우 날짜 시퀀스 준비
        dates = X.index.to_list()  # X가 DataFrame이면 OK. 아니면 df_agent.index 사용

        for sample_idx in range(samples):
            for time_idx in range(time_steps):
                row = {
                    'sample_id': sample_idx,
                    'time_step': time_idx,
                    'date' : str(dates[sample_idx + time_idx]),
                    'target': y_seq[sample_idx, 0] if time_idx == time_steps - 1 else np.nan,
                }
                for feat_idx, feat_name in enumerate(col):
                    row[feat_name] = X_seq[sample_idx, time_idx, feat_idx]
                flattened_data.append(row)

        agent_df = pd.DataFrame(flattened_data)
        csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
        agent_df.to_csv(csv_path, index=False)
        print(f"✅ {ticker} {agent_id} dataset saved to CSV ({len(X_seq)} samples, {len(col)} features)")


# --------------------------------------------
# CSV 데이터 로드 함수들
# --------------------------------------------

# 아연수정
def load_dataset(ticker, agent_id=None, save_dir=dir_info["data_dir"]):
    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 날짜 컬럼 datetime으로 변환 후 feature에서 제외
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 피처 컬럼 추출 (sample_id, time_step, target, date 제외)
    meta_cols = {"sample_id", "time_step", "target", "date"}
    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)
    ]

    # 시퀀스 데이터로 재구성
    unique_samples = df['sample_id'].nunique()
    time_steps = df['time_step'].nunique()
    n_features = len(feature_cols)

    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)

    dates_all = []

    for i, sample_id in enumerate(df['sample_id'].unique()):
        s = df[df["sample_id"] == sample_id].sort_values("time_step")
        X[i] = s[feature_cols].values
        y[i, 0] = s["target"].iloc[-1]
        if "date" in s.columns:
            dates_all.append(s["date"].dt.strftime("%Y-%m-%d").tolist())
        else:
            dates_all.append([None]*time_steps)

    return X, y, feature_cols, dates_all

def get_latest_close_price(ticker, save_dir=dir_info["data_dir"]):
    # 원본 데이터에서 최신 Close 가격 가져오기
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df['Close'].iloc[-1])
    else:
        # 원본 데이터가 없으면 yfinance로 직접 가져오기
        import yfinance as yf
        data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
        return float(data['Close'].iloc[-1])


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))