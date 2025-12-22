# core/data_set.py

from __future__ import annotations
import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from config.agents_set import agents_info, dir_info
from .technical_classes import technical_data_set as _techds

_HAS_MACRO = False
_MACRO_IMPORT_ERROR = ""
try:
    # 1) 패키지 내부 상대 임포트
    from .macro_classes.macro_funcs import macro_dataset as _macro_dataset
    macro_dataset = _macro_dataset
    _HAS_MACRO = True
    _MACRO_SRC = "relative: .macro_classes.macro_funcs"
except Exception as e1:
    try:
        # 2) 절대 임포트 (환경에 따라 상대가 막힐 때)
        from core.macro_classes.macro_funcs import macro_dataset as _macro_dataset
        macro_dataset = _macro_dataset
        _HAS_MACRO = True
        _MACRO_SRC = "absolute: core.macro_classes.macro_funcs"
    except Exception as e2:
        macro_dataset = None
        _HAS_MACRO = False
        _MACRO_IMPORT_ERROR = f"{type(e1).__name__}: {e1} | {type(e2).__name__}: {e2}"
        _MACRO_SRC = "unavailable"


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def create_sequences(features: pd.DataFrame, target: np.ndarray, window_size: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i + window_size].to_numpy())
        y.append(target[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

def _save_agent_csv(flattened_rows: List[dict], csv_path: str) -> None:
    agent_df = pd.DataFrame(flattened_rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    agent_df.to_csv(csv_path, index=False, encoding="utf-8")


def _fetch_ticker_data_for_sentimental(ticker: str, period: Optional[str], interval: Optional[str]) -> pd.DataFrame:
    period = period or "2y"
    interval = interval or "1d"

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df.dropna(inplace=True)

    # 멀티인덱스 컬럼 방어
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 기본 기술 지표
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

    # Fundamental 보조(USD/KRW, NASDAQ, VIX)
    try:
        usd_krw = yf.download("USDKRW=X", period=period, interval=interval, auto_adjust=False, progress=False)
        df["USD_KRW"] = (usd_krw["Close"].reindex(df.index, method="ffill") if not usd_krw.empty else 1300.0)

        nasdaq = yf.download("^IXIC", period=period, interval=interval, auto_adjust=False, progress=False)
        df["NASDAQ"] = (nasdaq["Close"].reindex(df.index, method="ffill") if not nasdaq.empty else 15000.0)

        vix = yf.download("^VIX", period=period, interval=interval, auto_adjust=False, progress=False)
        df["VIX"] = (vix["Close"].reindex(df.index, method="ffill") if not vix.empty else 20.0)
    except Exception as e:
        print(f"⚠️ 추가 지표 다운로드 실패: {e}")
        df["USD_KRW"] = 1300.0
        df["NASDAQ"] = 15000.0
        df["VIX"] = 20.0

    # 감성(placeholder): 초기 코드 그대로
    df["sentiment_mean"] = df["returns"].rolling(3).mean().fillna(0)
    df["sentiment_vol"] = df["returns"].rolling(3).std().fillna(0)

    df.dropna(inplace=True)
    return df


def build_dataset(
        ticker: str,
        save_dir: str = dir_info["data_dir"],
        agent_id: Optional[str] = None,
        period: Optional[str] = None,
        interval: Optional[str] = None,
) -> None:
    """
    debate_system.py에서 agent_id를 넘겨주면, 여기서 분기 처리
    - agent_id == 'MacroAgent' / '매크로' / 'macro'
    - agent_id == 'SentimentalAgent' / '센티멘탈' / 'sentimental'
    - agent_id == 'TechnicalAgent' / '테크니컬' / 'technical' (추후)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # agent_id가 지정되면 해당 에이전트만 처리, 없으면 모든 에이전트 처리
    agents_to_process = []
    if agent_id:
        # agent_id 정규화
        agent_id_norm = str(agent_id).lower()
        if agent_id_norm in {"macroagent", "macro", "매크로"}:
            agents_to_process = ["MacroAgent"]
        elif agent_id_norm in {"sentimentalagent", "sentimental", "센티멘탈"}:
            agents_to_process = ["SentimentalAgent"]
        elif agent_id_norm in {"technicalagent", "technical", "테크니컬"}:
            agents_to_process = ["TechnicalAgent"]
        else:
            # 정확히 일치하는 경우
            if agent_id in agents_info:
                agents_to_process = [agent_id]
            else:
                raise ValueError(f"지원하지 않는 agent_id: {agent_id}")
    else:
        # agent_id가 없으면 모든 에이전트 처리
        agents_to_process = list(agents_info.keys())
    
    # 공통 RAW는 테크니컬 전용 fetch로 통일 (TechnicalAgent가 처리 대상인 경우만)
    if "TechnicalAgent" in agents_to_process:
        raw = _techds.fetch_ticker_data(
            ticker,
            period=period or "5y",
            interval=interval or "1d",
        )
        raw.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True)

    # Agent별 데이터셋을 CSV로 저장
    for aid in agents_to_process:
        # ---------- macro_agent ----------
        if aid == "MacroAgent":
            if not _HAS_MACRO or macro_dataset is None:
                raise ImportError(
                    "macro_dataset 모듈을 찾을 수 없습니다. core/macro_classes 확인 필요 "
                    f"details={_MACRO_IMPORT_ERROR}"
                )
            macro_dataset(ticker_name=ticker)
            print(f"✅ {ticker} MacroAgent dataset saved (macro_dataset 호출 via {_MACRO_SRC})")

        # ---------- sentimental_agent ----------
        elif aid == "SentimentalAgent":
            df = _fetch_ticker_data_for_sentimental(ticker, period, interval)

            # 원본 CSV 저장(후속처리 참고용)
            df.to_csv(os.path.join(save_dir, f"{ticker}_raw_data.csv"), index=True, encoding="utf-8")

            # 사용할 피처 컬럼 (SentimentalAgent의 FEATURE_COLS 사용)
            if aid in agents_info:
                # config에서 window_size 가져오기
                window_size = agents_info[aid].get("window_size", 40)
                # SentimentalAgent는 FEATURE_COLS를 사용하므로 직접 정의
                feature_cols = [
                    "return_1d",
                    "hl_range",
                    "Volume",
                    "news_count_1d",
                    "news_count_7d",
                    "sentiment_mean_1d",
                    "sentiment_mean_7d",
                    "sentiment_vol_7d",
                ]
            else:
                # fallback: 기본 피처
                feature_cols = [
                    "returns", "sma_5", "sma_20", "rsi", "volume_z",
                    "USD_KRW", "NASDAQ", "VIX",
                    "sentiment_mean", "sentiment_vol",
                    "Open", "High", "Low", "Close", "Volume",
                ]
                window_size = 40

            # 타깃: 다음날 수익률
            returns = df["Close"].pct_change().shift(-1)
            valid_mask = ~returns.isna()
            y = returns[valid_mask].to_numpy().reshape(-1, 1)
            
            # feature_cols가 df에 없는 경우를 대비해 존재하는 컬럼만 사용
            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < len(feature_cols):
                print(f"[WARN] 일부 피처 컬럼이 없습니다. 사용 가능한 컬럼: {available_cols}")
                # 기본 피처로 대체
                feature_cols = ["returns", "Close", "Volume", "sentiment_mean", "sentiment_vol"]
                available_cols = [c for c in feature_cols if c in df.columns]
            
            X = df.loc[valid_mask, available_cols]

            # 시퀀스 생성
            X_seq, y_seq = create_sequences(X, y, window_size=window_size)
            samples, time_steps, n_feats = X_seq.shape
            print(f"[SentimentalAgent] X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

            # 날짜 정보 준비
            dates = X.index

            # 플랫 CSV
            flattened = []
            for sample_idx in range(samples):
                # 윈도우 내 날짜들
                window_dates = dates[sample_idx : sample_idx + time_steps]
                
                for time_idx in range(time_steps):
                    row = {
                        "sample_id": sample_idx,
                        "time_step": time_idx,
                        "target": float(y_seq[sample_idx, 0]) if time_idx == time_steps - 1 else np.nan,
                        "date": window_dates[time_idx].strftime("%Y-%m-%d"),
                    }
                    for feat_idx, feat_name in enumerate(available_cols):
                        row[feat_name] = float(X_seq[sample_idx, time_idx, feat_idx])
                    flattened.append(row)

            csv_path = os.path.join(save_dir, f"{ticker}_{aid}_dataset.csv")
            _save_agent_csv(flattened, csv_path)
            print(f"✅ {ticker} {aid} dataset saved to CSV ({samples} samples, {len(available_cols)} features)")

        # ---------- technical_agent ----------
        elif aid == "TechnicalAgent":
            # common_params에서 period 가져오기
            from config.agents_set import common_params
            period_to_use = period or common_params.get("period", "2y")
            _techds.build_dataset(
                ticker=ticker,
                save_dir=save_dir,
                period=period_to_use,
                interval=interval or agents_info["TechnicalAgent"].get("interval", "1d"),
            )
            print(f"✅ {ticker} TechnicalAgent dataset saved via technical_data_set")

        else:
            raise ValueError(f"지원하지 않는 agent_id: {aid}")


def load_dataset(ticker: str, agent_id: str, save_dir: str = dir_info["data_dir"], return_dates: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    위에서 저장한 CSV({ticker}_{agent_id}_dataset.csv)를 다시 시퀀스로 복원
    - 숫자형 컬럼만 사용하도록 안전 가드 추가(날짜/문자열 혼입 방지)
    - return_dates=True일 경우 (X, y, feature_cols, dates_all) 반환
    """
    # 1) 테크니컬은 전용 로더로 위임
    norm = str(agent_id).lower()
    if norm in {"technicalagent", "technical", "테크니컬"}:
        X, y, feature_cols, _dates = _techds.load_dataset(
            ticker=ticker,
            agent_id="TechnicalAgent",
            save_dir=save_dir,
        )
        if return_dates:
            return X, y, feature_cols, _dates
        return X, y, feature_cols

    csv_path = os.path.join(save_dir, f"{ticker}_{agent_id}_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 후보 피처: 플랫 CSV 기준 기본 제외 컬럼
    candidate_cols = [c for c in df.columns if c not in ["sample_id", "time_step", "target", "date"]]

    unique_samples = df["sample_id"].nunique()
    time_steps = df["time_step"].nunique()

    # 최초 블록(가장 작은 sample_id)에서 숫자형 컬럼만 확정
    first_id = sorted(df["sample_id"].unique())[0]
    first_block = df[df["sample_id"] == first_id].sort_values("time_step")[candidate_cols]

    numeric_feature_cols: List[str] = []
    for c in candidate_cols:
        s = first_block[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric_feature_cols.append(c)

    dropped = [c for c in candidate_cols if c not in numeric_feature_cols]
    if dropped:
        print(f"[warn] Non-numeric features dropped in load_dataset(): {dropped}")

    feature_cols = numeric_feature_cols
    n_features = len(feature_cols)

    if n_features == 0:
        raise ValueError("No numeric feature columns found after filtering. Check your dataset builder.")

    X = np.zeros((unique_samples, time_steps, n_features), dtype=np.float32)
    y = np.zeros((unique_samples, 1), dtype=np.float32)
    
    dates_all = []
    has_date = "date" in df.columns

    for i, sample_id in enumerate(sorted(df["sample_id"].unique())):
        block = df[df["sample_id"] == sample_id].sort_values("time_step")

        # 안전을 위해 숫자형 변환 강제 시도(실패 시 NaN → 이후 astype에서 에러 방지)
        block_numeric = block[feature_cols].apply(pd.to_numeric, errors="coerce")
        if block_numeric.isna().any().any():
            # 남은 NaN은 직전값 채우고, 그래도 남으면 0으로
            block_numeric = block_numeric.fillna(method="ffill").fillna(0.0)

        X[i] = block_numeric.to_numpy(dtype=np.float32)

        # 마지막 타임스텝에만 target 값이 들어가 있으므로 그 값을 사용
        y_val = block["target"].dropna()
        y[i, 0] = float(y_val.iloc[-1]) if not y_val.empty else np.nan
        
        if return_dates:
            if has_date:
                dates_all.append(block["date"].tolist())
            else:
                dates_all.append([None] * time_steps)

    if return_dates:
        return X, y, feature_cols, dates_all
        
    return X, y, feature_cols


def get_latest_close_price(ticker: str, save_dir: str = dir_info["data_dir"]) -> float:
    raw_data_path = os.path.join(save_dir, f"{ticker}_raw_data.csv")
    if os.path.exists(raw_data_path):
        df = pd.read_csv(raw_data_path, index_col=0)
        return float(df["Close"].iloc[-1])
    data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
    return float(data["Close"].iloc[-1])


def build_targets(close: pd.Series) -> pd.Series:
    # P_{t+1} / P_t - 1
    ret = close.shift(-1) / close - 1.0
    # 마지막 행은 타깃 없음 → 제거
    return ret.iloc[:-1]

def build_features(df: pd.DataFrame, window: int):
    # df.index는 날짜 정렬 가정, 마지막 행 제외하고 X, y 정렬 맞추기
    y = build_targets(df["Close"])
    X = df.iloc[:-1]          # y와 길이 정합
    return X, y
