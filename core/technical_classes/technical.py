# core/technical_classes/technical.py
# ===============================================
# Technical Feature 데이터 수집
# ===============================================
import numpy as np
import pandas as pd

"""
테크니컬 지표(13개)
"""

TECH_COLS = [
    "weekofyear_sin","weekofyear_cos","log_ret_lag1",
    "ret_3d","mom_10","ma_200",
    "macd","bbp","adx_14",
    "obv","vol_ma_20","vol_chg","vol_20d",
    ]

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _bollinger_pb(s, p=20, k=2.0, eps=1e-12):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    up = m + k*sd
    lo = m - k*sd
    denom = (up - lo).replace(0.0, np.nan)
    return ((s - lo) / (denom + eps))

def _momentum(s, n):
    return s - s.shift(n)

def _true_range(h, l, c):
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr

def _adx(h, l, c, n=14):
    # pandas Series → 1D ndarray로 정규화
    up = h.diff().to_numpy().reshape(-1)     # (N,)
    dn = (-l.diff()).to_numpy().reshape(-1)  # (N,)

    # 1D에서 연산 수행
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)      # (N,)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)      # (N,)

    # TR/ATR는 Series 계산을 유지해서 인덱스 정합 보장
    tr = _true_range(h, l, c)                               # pd.Series (N,)
    atr_n = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di  = 100 * pd.Series(plus_dm,  index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (denom + 1e-12)

    return dx.ewm(alpha=1/n, adjust=False).mean()

def _obv(c, v):
    direction = np.sign(c.diff().fillna(0.0))
    return (direction * v.fillna(0.0)).cumsum()

def build_features_technical(df_price: pd.DataFrame) -> pd.DataFrame:
    """
    입력: OHLCV DataFrame[Open,High,Low,Close,Volume]
    출력: TECH 12개 DataFrame(float32), 인덱스 동일
    요구 최소 길이: max(200, 20+p) 권장
    """
    o,h,l,c,v = df_price["Open"],df_price["High"],df_price["Low"],df_price["Close"],df_price["Volume"]
    out = pd.DataFrame(index=df_price.index)

    # 주차 순환 인코딩(연속성 보장)
    week = df_price.index.isocalendar().week.astype(float)
    out["weekofyear_sin"] = np.sin(2 * np.pi * week / 52.0)
    out["weekofyear_cos"] = np.cos(2 * np.pi * week / 52.0)

    # r2 -> log_ret_lag1 변경
    out["log_ret_lag1"] = np.log(c.shift(1) / c.shift(2))

    out["ret_3d"] = c.pct_change(3)
    out["mom_10"] = _momentum(c, 10)
    out["ma_200"] = c.rolling(200).mean()

    ema12, ema26 = _ema(c,12), _ema(c,26)
    out["macd"] = ema12 - ema26
    out["bbp"] = _bollinger_pb(c,20,2.0)
    out["adx_14"] = _adx(h,l,c,14)
    out["obv"] = _obv(c,v)
    out["vol_ma_20"] = v.rolling(20).mean()
    out["vol_chg"] = v.pct_change(1)

    ret_1d = c.pct_change(1)
    out["vol_20d"] = ret_1d.rolling(20).std()

    # 수치화→이상치 처리→ffill→최종 dropna→float32
    out = (out.apply(pd.to_numeric, errors="coerce")
              .replace([np.inf,-np.inf], np.nan)
              .ffill()
              .dropna()
              .astype(np.float32))
    
    out = out.reindex(columns=TECH_COLS).astype(np.float32)
    return out
