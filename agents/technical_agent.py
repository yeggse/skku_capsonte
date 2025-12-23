# agents/technical_agent.py

import os
import json
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

from agents.base_agent import (
    BaseAgent, StockData, Target, Opinion, Rebuttal
)

from config.agents_set import agents_info, dir_info, common_params
from config.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from core.base_classes.base_predict import BaseAgentPredictMixin
from core.technical_classes.technical_explainer import TechnicalExplainer
from core.technical_classes.technical_data_set import load_dataset

import warnings

from core.technical_classes.technical_trainer import TechnicalTrainer

warnings.filterwarnings('ignore')


def round_num(value: Union[float, int, str], decimals: int = 4) -> float:
    """숫자 값을 지정된 소수점 자리수로 반올림하여 반환합니다"""
    try:
        return float(f"{float(value):.{decimals}f}")
    except (ValueError, TypeError):
        return value

class TechnicalAgent(BaseAgent, nn.Module, BaseAgentPredictMixin):
    """
    기술적 분석 기반 주가 예측 에이전트
    
    주가 차트 데이터를 분석하여 주가 예측을 수행합니다.
    """

    def __init__(self,
        agent_id="TechnicalAgent",
        input_dim=agents_info["TechnicalAgent"]["input_dim"],
        rnn_units1=agents_info["TechnicalAgent"]["rnn_units1"],
        rnn_units2=agents_info["TechnicalAgent"]["rnn_units2"],
        dropout=agents_info["TechnicalAgent"]["dropout"],
        data_dir=dir_info["data_dir"],
        window_size=agents_info["TechnicalAgent"]["window_size"],
        epochs=agents_info["TechnicalAgent"]["epochs"],
        learning_rate=agents_info["TechnicalAgent"]["learning_rate"],
        batch_size=agents_info["TechnicalAgent"]["batch_size"],
        **kwargs
    ):
        """TechnicalAgent 초기화"""
        nn.Module.__init__(self)
        BaseAgent.__init__(self, agent_id=agent_id, data_dir=data_dir, **kwargs)

        self.input_dim = int(input_dim)
        self.u1         = int(rnn_units1)
        self.u2         = int(rnn_units2)
        self.window_size= int(window_size)
        self.epochs     = int(epochs)
        self.lr         = float(learning_rate)
        self.batch_size = int(batch_size)

        self.lstm1 = nn.LSTM(self.input_dim, self.u1, batch_first=True)
        self.lstm2 = nn.LSTM(self.u1, self.u2, batch_first=True)
        self.attn_vec = nn.Parameter(torch.randn(self.u2))
        self.fc = nn.Linear(self.u2, 1)
        self.drop = nn.Dropout(float(dropout))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        huber_delta = common_params.get("huber_loss_delta", 1.0)
        self.loss_fn = nn.HuberLoss(delta=huber_delta)
        
        self.last_pred = None
        self.last_attn = None
        self._last_idea = None

        self.explainer = TechnicalExplainer(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """모델 Forward Pass"""
        h1, _ = self.lstm1(x)
        h1 = self.drop(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop(h2)

        w = torch.softmax(torch.matmul(h2, self.attn_vec), dim=1)
        self._last_attn = w.detach()
        ctx = (h2 * w.unsqueeze(-1)).sum(dim=1)
        return self.fc(ctx)


    def _build_messages_opinion(self, stock_data, target):
        """Opinion 생성용 프롬프트 메시지를 구성합니다"""
        last = float(getattr(stock_data, "last_price", target.next_close))
        agent_data = getattr(stock_data, self.agent_id, {})
        
        if isinstance(agent_data, dict) and agent_data:
            df = pd.DataFrame(agent_data)
            X_last = torch.tensor(
                df.tail(self.window_size).values, 
                dtype=torch.float32
            ).unsqueeze(0)
        else:
            print(f"[WARN] {self.agent_id} stockdata가 비어있음, searcher 재호출")
            X_last = self.search(self.ticker)
            if not isinstance(X_last, torch.Tensor):
                X_last = torch.tensor(X_last, dtype=torch.float32)
        
        dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])
        cfg = agents_info.get(self.agent_id, {})
        top_k = cfg.get("top_k_features", 5)


        exp = self.explainer.explain(
            X_last=X_last,
            dates=dates,
            feature_names=self.stockdata.feature_cols,
            top_k=top_k,
            use_shap=True
        )

        idea = self.explainer.summarize(exp)
        self._last_idea = idea

        ctx = {
            "ticker": getattr(stock_data, "ticker", "Unknown"),
            "last_price": round_num(last),
            "next_close": round_num(target.next_close),
            "uncertainty": round_num(target.uncertainty),
            "confidence": round_num(target.confidence),
            "sigma": round_num(target.uncertainty or 0.0),
            "beta": round_num(target.confidence or 0.0),
            "window_size": int(self.window_size),
            "idea": idea,
        }

        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        tmpl = OPINION_PROMPTS[self.agent_id]["user"]
        user_text = tmpl.replace("{context}", json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_rebuttal(self, my_opinion: Opinion, target_opinion: Opinion, stock_data: StockData) -> tuple[str, str]:
        """Rebuttal 생성용 프롬프트 메시지를 구성합니다"""
        t = stock_data.ticker or "UNKNOWN"
        ccy = (stock_data.currency or "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "data_summary": getattr(stock_data, "feature_cols", []),
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": str(my_opinion.reason)[:2000],
                "uncertainty": self._safe_float(my_opinion.target.uncertainty, 0.0),
                "confidence": self._safe_float(my_opinion.target.confidence, 0.5),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": self._safe_float(target_opinion.target.uncertainty, 0.0),
                "confidence": self._safe_float(target_opinion.target.confidence, 0.5),
            },

        }
    
        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-self.window_size:]
            else:
                ctx[col] = [values]

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        tmpl = REBUTTAL_PROMPTS[self.agent_id]["user"]
        user_text = tmpl.replace("{context}", json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_revision(self, my_opinion: Opinion, others: List[Opinion], rebuttals: Optional[List[Rebuttal]] = None, stock_data: StockData = None) -> tuple[str, str]:
        """Revision 생성용 프롬프트 메시지를 구성합니다"""
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD").upper()
        agent_data = getattr(stock_data, self.agent_id, None)
        if not agent_data or not isinstance(agent_data, dict):
            raise ValueError(f"{self.agent_id} 데이터 구조 오류: dict형 컬럼 데이터가 필요함")

        others_summary = []
        for o in others:
            entry = {
                "agent_id": o.agent_id,
                "predicted_price": float(o.target.next_close),
                "confidence": float(o.target.confidence),
                "uncertainty": float(o.target.uncertainty),
                "reason": str(o.reason)[:500],
            }
            if rebuttals:
                related_rebuts = [
                    {"stance": r.stance, "message": r.message}
                    for r in rebuttals
                    if r.from_agent_id == o.agent_id and r.to_agent_id == self.agent_id
                ]
                if related_rebuts:
                    entry["rebuttals_to_me"] = related_rebuts
            others_summary.append(entry)

        ctx = {
            "ticker": t,
            "currency": ccy,
            "agent_type": self.agent_id,
            "my_opinion": {
                "predicted_price": float(my_opinion.target.next_close),
                "uncertainty": (
                    float(my_opinion.target.uncertainty)
                    if my_opinion.target.uncertainty is not None
                    else 0.0
                ),
                "confidence": (
                    float(my_opinion.target.confidence)
                    if my_opinion.target.confidence is not None
                    else 0.5
                ),

                "reason": str(my_opinion.reason)[:1000],
            },
            "others_summary": others_summary,
            "data_summary": getattr(stock_data, "feature_cols", []),
        }

        for col, values in agent_data.items():
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-14:]  # 최근 14일치
            else:
                ctx[col] = [values]

        prompt_set = REVISION_PROMPTS.get(self.agent_id)
        system_text = prompt_set["system"]
        user_text = prompt_set["user"].format(context=json.dumps(ctx, ensure_ascii=False, indent=2))
        return system_text, user_text

    def _fetch_ticker_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """yfinance로 데이터를 다운로드하고 기본 지표를 계산합니다"""
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        df.dropna(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df["returns"] = df["Close"].pct_change().fillna(0)
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["rsi"] = self._compute_rsi(df["Close"])
        df["volume_z"] = (df["Volume"] - df["Volume"].mean()) / (df["Volume"].std() + 1e-6)

        df.dropna(inplace=True)
        return df

    def _compute_rsi(self, series, window=14):
        """RSI를 계산합니다"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / (avg_loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def _create_sequences(self, features, target, window_size):
        """시계열 시퀀스 데이터를 생성합니다"""
        X, y = [], []
        for i in range(len(features) - window_size):
            X.append(features[i:i + window_size])
            y.append(target[i + window_size])
        return np.array(X), np.array(y)

    def _build_features_technical(self, df_price: pd.DataFrame) -> pd.DataFrame:
        """테크니컬 피처를 생성합니다"""
        o, h, l, c, v = df_price["Open"], df_price["High"], df_price["Low"], df_price["Close"], df_price["Volume"]
        out = pd.DataFrame(index=df_price.index)
        
        def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
        
        def _bollinger_pb(s, p=20, k=2.0, eps=1e-12):
            m = s.rolling(p).mean()
            sd = s.rolling(p).std()
            up = m + k*sd
            lo = m - k*sd
            denom = (up - lo).replace(0.0, np.nan)
            return ((s - lo) / (denom + eps))
        
        def _momentum(s, n): return s - s.shift(n)
        
        def _true_range(h, l, c):
            tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
            return tr
        
        def _adx(h, l, c, n=14):
            up = h.diff().to_numpy().reshape(-1)
            dn = (-l.diff()).to_numpy().reshape(-1)
            plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
            minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
            tr = _true_range(h, l, c)
            atr_n = tr.ewm(alpha=1/n, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
            minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / atr_n
            denom = (plus_di + minus_di).replace(0, np.nan)
            dx = 100 * (plus_di - minus_di).abs() / (denom + 1e-12)
            return dx.ewm(alpha=1/n, adjust=False).mean()
        
        def _obv(c, v):
            direction = np.sign(c.diff().fillna(0.0))
            return (direction * v.fillna(0.0)).cumsum()
        
        week = df_price.index.isocalendar().week.astype(float)
        out["weekofyear_sin"] = np.sin(2 * np.pi * week / 52.0)
        out["weekofyear_cos"] = np.cos(2 * np.pi * week / 52.0)
        out["log_ret_lag1"] = np.log(c.shift(1) / c.shift(2))
        out["ret_3d"] = c.pct_change(3)
        out["mom_10"] = _momentum(c, 10)
        out["ma_200"] = c.rolling(200).mean()
        
        ema12, ema26 = _ema(c, 12), _ema(c, 26)
        out["macd"] = ema12 - ema26
        out["bbp"] = _bollinger_pb(c, 20, 2.0)
        out["adx_14"] = _adx(h, l, c, 14)
        out["obv"] = _obv(c, v)
        out["vol_ma_20"] = v.rolling(20).mean()
        out["vol_chg"] = v.pct_change(1)
        
        ret_1d = c.pct_change(1)
        out["vol_20d"] = ret_1d.rolling(20).std()
        
        out = (out.apply(pd.to_numeric, errors="coerce")
                  .replace([np.inf, -np.inf], np.nan)
                  .ffill()
                  .dropna()
                  .astype(np.float32))
        
        tech_cols = [
            "weekofyear_sin", "weekofyear_cos", "log_ret_lag1",
            "ret_3d", "mom_10", "ma_200",
            "macd", "bbp", "adx_14",
            "obv", "vol_ma_20", "vol_chg", "vol_20d"
        ]
        out = out.reindex(columns=tech_cols).astype(np.float32)
        return out



    def search(self, ticker: Optional[str] = None, rebuild: bool = False):
        agent_id = self.agent_id
        self.ticker = ticker or self.ticker
        if not self.ticker:
            raise ValueError("ticker가 지정되지 않았습니다.")

        raw_csv_path = self._resolve_raw_csv_path()

        if rebuild or not os.path.exists(raw_csv_path):
            self._ensure_raw_csv(raw_csv_path)

        df_raw = self._load_raw_csv(raw_csv_path)
        df_raw = self._validate_feature_columns(df_raw)

        x_latest, dates = self._build_latest_window(df_raw)
        self._build_stockdata(df_raw, x_latest, dates)

        print(f"✅ [{agent_id}] Searcher 완료: 윈도우 shape {x_latest.shape}")
        return torch.tensor(x_latest, dtype=torch.float32)

    def _resolve_raw_csv_path(self) -> str:
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        os.makedirs(raw_dir, exist_ok=True)

        base_path = os.path.join(raw_dir, f"{self.ticker}_{self.agent_id}_raw.csv")

        if getattr(self, "test_mode", False) and getattr(self, "simulation_date", None):
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{self.ticker}_{self.agent_id}_raw_{date_str}.csv")
            if os.path.exists(temp_path):
                print(f"[INFO] 백테스트 모드: {self.simulation_date} 이전 데이터 사용")
                return temp_path

        return base_path

    def _ensure_raw_csv(self, raw_csv_path: str):
        print(f"[{self.agent_id}] Raw CSV 생성 중...")
        cfg = agents_info.get(self.agent_id, {})
        interval = cfg.get("interval", "1d")

        period = common_params.get("period", "2y")
        df_price = self._fetch_ticker_data(self.ticker, period, interval)

        feat_df = self._build_features_technical(
            df_price[["Open", "High", "Low", "Close", "Volume"]]
        )

        self._save_raw_csv(df_price, feat_df, raw_csv_path)

    def _save_raw_csv(self, df_price: pd.DataFrame, feat_df: pd.DataFrame, path: str):
        df = feat_df.copy()
        df.index.name = "Date"
        df.reset_index(inplace=True)

        close_df = df_price[["Close"]].copy()
        close_df.index.name = "Date"
        close_df.reset_index(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        close_df["Date"] = pd.to_datetime(close_df["Date"])

        df = df.merge(close_df, on="Date", how="left")
        df["ticker"] = self.ticker

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

        print(f"✅ Raw CSV 저장 완료: {path} ({len(df):,} rows)")

    def _load_raw_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw CSV not found: {path}")

        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)

    def _validate_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = agents_info.get(self.agent_id, {})
        feature_cols = cfg.get("data_cols", [])

        if not feature_cols:
            raise ValueError(f"[{self.agent_id}] config에 data_cols가 정의되지 않았습니다.")

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"[WARN] 누락된 feature {len(missing)}개 → 0.0으로 보정")
            for c in missing:
                df[c] = 0.0

        return df

    def _build_latest_window(self, df: pd.DataFrame):
        cfg = agents_info[self.agent_id]
        window_size = cfg["window_size"]
        feature_cols = cfg["data_cols"]

        X_all = df[feature_cols].values.astype(np.float32)
        if len(X_all) < window_size:
            raise ValueError("데이터 길이가 window_size보다 짧습니다.")

        x_latest = X_all[-window_size:].reshape(1, window_size, -1)
        dates = df["Date"].iloc[-window_size:].astype(str).tolist()
        return x_latest, dates

    def _build_stockdata(self, df: pd.DataFrame, x_latest: np.ndarray, dates: list):
        sd = StockData(ticker=self.ticker)
        sd.feature_cols = agents_info[self.agent_id]["data_cols"]
        sd.window_size = len(dates)
        sd.last_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else None

        try:
            sd.currency = yf.Ticker(self.ticker).info.get("currency", "USD")
        except Exception:
            sd.currency = "USD"

        df_latest = pd.DataFrame(x_latest[0], columns=sd.feature_cols)
        setattr(sd, self.agent_id, {c: df_latest[c].tolist() for c in df_latest.columns})
        setattr(sd, f"{self.agent_id}_dates", dates)

        self.stockdata = sd



    def pretrain(self):
        if not self.ticker:
            raise ValueError("ticker 미설정")

        raw_csv_path = self._resolve_raw_csv_path()
        df_raw = self._load_raw_csv(raw_csv_path)

        trainer = TechnicalTrainer(self)
        X, y = trainer.prepare_dataset(df_raw)   # ← 여기서 X, y가 시퀀스
        trainer.fit(X, y)
        trainer.save()

        self.model_loaded = True

        # ===============================
        # dataset.csv 저장 (앙상블용)
        # ===============================
        if common_params.get("pretrain_save_dataset", True):
            dataset_path = os.path.join(
                self.data_dir,
                f"{self.ticker}_{self.agent_id}_dataset.csv"
            )

            rows = []
            for sample_id in range(len(X)):
                for t in range(self.window_size):
                    row = {
                        "sample_id": sample_id,
                        "time_step": t,
                        "target": (
                            float(y[sample_id])
                            if t == self.window_size - 1
                            else np.nan
                        ),
                    }
                    for i, col in enumerate(self.stockdata.feature_cols):
                        row[col] = float(X[sample_id, t, i])
                    rows.append(row)

            df_dataset = pd.DataFrame(rows)
            os.makedirs(self.data_dir, exist_ok=True)
            df_dataset.to_csv(dataset_path, index=False)
            print(f"✅ TechnicalAgent dataset 저장 완료: {dataset_path}")

    def review_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """초기 의견을 생성합니다"""
        if stock_data is not None:
            self.stockdata = stock_data
        else:
            if getattr(self, "stockdata", None) is None:
                if not self.ticker:
                    raise RuntimeError("ticker 미설정")
                _ = self.search(self.ticker)
            stock_data = self.stockdata

        if target is None:
            agent_data = getattr(stock_data, self.agent_id, {})
            if isinstance(agent_data, dict) and agent_data:
                df = pd.DataFrame(agent_data)
                X_input = torch.tensor(
                    df.tail(self.window_size).values, 
                    dtype=torch.float32
                ).unsqueeze(0)
            else:
                X_input = self.search(self.ticker)
            target = self.predict(X_input)

        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object", 
                "properties": {"reason": {"type": "string"}}, 
                "required": ["reason"], 
                "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")
        self.opinions.append(Opinion(
                    agent_id=self.agent_id, 
                    target=target, 
                    reason=reason))
        return self.opinions[-1]

    def review_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """반박을 생성합니다"""
        sys_text, user_text = self._build_messages_rebuttal(
            my_opinion=my_opinion,
            target_opinion=other_opinion,
            stock_data=self.stockdata
        )

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {
                    "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                    "message": {"type": "string"},
                    "support_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "지지율 (0~1). SUPPORT일 때만 유효, REBUT일 때는 0"
                    }
                },
                "required": ["stance", "message", "support_rate"],
                "additionalProperties": False
            }
        )

        stance = parsed.get("stance", "REBUT")
        
        # STANCE에 따라 support_rate 설정
        if stance == "SUPPORT":
            # SUPPORT일 때는 0~1 사이의 지지율 입력
            support_rate = parsed.get("support_rate")
            if support_rate is None:
                support_rate = 0.5  # 기본값
            # 0~1 범위로 클리핑
            support_rate = max(0.0, min(1.0, float(support_rate)))
        else:
            # REBUT일 때는 0으로 설정
            support_rate = 0.0

        result = Rebuttal(
            from_agent_id=my_opinion.agent_id,
            to_agent_id=other_opinion.agent_id,
            stance=stance,
            message=parsed.get("message", "(반박/지지 사유 생성 실패)"),
            support_rate=support_rate
        )

        self.rebuttals[round].append(result)
        if self.verbose:
            print(f"[{self.agent_id}] rebuttal 생성 → {result.stance}, support_rate: {support_rate}")
        return result
    
    def reviewer_rebuttal(self, my_opinion, other_opinion, round_index):
        """호환용 래퍼 메서드"""
        return self.review_rebut(my_opinion, other_opinion, round_index)

    def review_revise(self, my_opinion, others, rebuttals, stock_data, fine_tune=True, lr=None, epochs=None):
        """의견을 수정하고 재예측합니다"""
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", 20)
        
        return super().review_revise(my_opinion, others, rebuttals, stock_data, fine_tune, lr, epochs)

    def load_model(self, model_path: Optional[str] = None):
        """모델을 로드합니다"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        try:
            # GPU 사용 가능 시 GPU로, 아니면 CPU로 로드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device)

            if isinstance(checkpoint, torch.nn.Module):
                state_dict = checkpoint.state_dict()
            elif isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("model_state_dict")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
            else:
                print(f"[{self.agent_id}] 알 수 없는 체크포맷: {type(checkpoint)}")
                return False

            self.load_state_dict(state_dict)
            # 모델을 GPU로 이동
            self.to(device)
            self.eval()
            self.model_loaded = True
            print(f"[{self.agent_id}] 모델 로드 완료: {model_path} (device: {device})")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] load_model 실패: {e}")
            return False

    def evaluate(self, ticker: str = None):
        """모델을 검증합니다"""
        if ticker is None:
            ticker = self.ticker

        X, y, feature_cols, _ = load_dataset(
            ticker,
            agent_id=self.agent_id,
            save_dir=self.data_dir
        )

        split_ratio = common_params.get("eval_split_ratio", 0.8)
        split_idx = int(len(X) * split_ratio)
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        self.scaler.load(ticker)
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_val_scaled = (y_val * y_scale_factor).reshape(-1)

        X_val_scaled, y_val_scaled = self.scaler.transform(X_val, y_val_scaled)
        y_val_scaled = np.asarray(y_val_scaled).reshape(-1)

        model_path = os.path.join(self.model_dir, f"{ticker}_{self.agent_id}.pt")
        if not self.load_model(model_path):
            self.pretrain()
            self.load_model(model_path)

        model = self
        model.eval()

        predictions = []
        actual_returns = []

        with torch.no_grad():
            for i in range(len(X_val_scaled)):
                X_input = X_val_scaled[i:i+1]
                X_tensor = torch.tensor(X_input, dtype=torch.float32)
                pred_scaled = model(X_tensor).item()
                predictions.append(pred_scaled)
                actual_returns.append(float(y_val_scaled[i]))

        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        mae = np.mean(np.abs(predictions - actual_returns))
        rmse = np.sqrt(np.mean((predictions - actual_returns) ** 2))

        if np.std(predictions) == 0 or np.std(actual_returns) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(predictions, actual_returns)[0, 1])

        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual_returns)
        direction_accuracy = float(np.mean(pred_direction == actual_direction) * 100.0)

        return {
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "direction_accuracy": direction_accuracy,
            "n_samples": len(predictions),
        }

    def _safe_float(self, v, default):
        return float(v) if v is not None else default
