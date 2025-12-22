# agents/technical_agent.py

import os
import json
from typing import List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

from agents.base_agent import (
    BaseAgent, StockData, Target, Opinion, Rebuttal
)

from core.technical_classes.technical_data_set import (
    load_dataset as load_dataset_tech,
)

from config.agents_set import agents_info, dir_info, common_params
from config.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

from core.technical_classes.technical_data_set import load_dataset
import shap

import warnings
warnings.filterwarnings('ignore')


def round_num(value: Union[float, int, str], decimals: int = 4) -> float:
    """숫자 값을 지정된 소수점 자리수로 반올림하여 반환합니다"""
    try:
        return float(f"{float(value):.{decimals}f}")
    except (ValueError, TypeError):
        return value

class TechnicalAgent(BaseAgent, nn.Module):
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
    
    def _validate_feature_names(self, feature_columns: List[str], num_features: int) -> List[str]:
        """피처 이름 리스트를 모델 입력 차원에 맞춰 보정합니다"""
        cols = list(feature_columns) if feature_columns else []
        if len(cols) != num_features:
            cols = cols[:num_features] + [f"f{i}" for i in range(len(cols), num_features)]
        return cols

    def _validate_dates(self, dates: List, window_size: int) -> List[str]:
        """날짜 리스트를 윈도우 길이에 맞춰 보정합니다"""
        if not dates or len(dates) != window_size:
            return [f"t-{window_size-1-i}" for i in range(window_size)]
        return [str(d) for d in dates]

    def _scale_like_train(self, X_np):
        """학습 시 사용한 스케일러로 입력을 스케일링합니다"""
        try:
            out = self.scaler.transform(X_np)
            if isinstance(out, tuple) and len(out) >= 1:
                return out[0]
            return out
        except Exception:
            return X_np

    @torch.no_grad()
    def time_importance_from_attention(self, X_last: torch.Tensor) -> np.ndarray:
        """모델의 Time-Attention 가중치를 기반으로 시간 중요도를 계산합니다"""
        self.eval()
        _ = self(X_last)
        attn = getattr(self, "_last_attn", None)

        if attn is None:
            T = X_last.shape[1]
            return np.ones(T, dtype=float) / T
        w = attn[0].abs().cpu().numpy()
        w = w.flatten() if w.ndim > 1 else w
        s = w.sum()
        result = w / s if s > 0 else np.ones_like(w) / len(w)
        return result.flatten() if result.ndim > 1 else result

    def gradxinput_attrib(self, X_last: torch.Tensor, eps: float = 0.0):
        """Grad×Input 방식으로 피처 기여도를 산출합니다"""
        self.eval()
        x = X_last.clone().detach().to(next(self.parameters()).device)

        if eps > 0:
            x = x + eps * torch.randn_like(x)
        x.requires_grad_(True)
        y = self(x).sum()
        self.zero_grad(set_to_none=True)
        y.backward()
        gi = (x.grad * x).abs()[0].detach().cpu().numpy()
        per_time = gi.sum(axis=1)
        per_feat = gi.mean(axis=0)
        return per_time, per_feat, gi

    @torch.no_grad()
    def occlusion_time(self, X_last: torch.Tensor, fill: str = "zero", batch: Optional[int] = None):
        """시간축 Occlusion을 통한 중요도를 계산합니다"""
        if batch is None:
            batch = agents_info.get(self.agent_id, {}).get("occlusion_batch_size", 32)
        self.eval()
        base = float(self(X_last).item())
        _, T, F = X_last.shape
        Xs = []
        for t in range(T):
            x = X_last.clone()
            if fill == "zero":
                x[:, t, :] = 0
            else:
                x[:, t, :] = X_last.mean(dim=1, keepdim=True)[:, 0, :]
            Xs.append(x)
        deltas = []
        for i in range(0, T, batch):
            xb = torch.cat(Xs[i:i+batch], dim=0)
            yb = self(xb).flatten().cpu().numpy()
            deltas.extend(np.abs(yb - base).tolist())
        s = sum(deltas)
        return np.array([v/s if s > 0 else 1.0/T for v in deltas], dtype=float)

    @torch.no_grad()
    def occlusion_feature(self, X_last: torch.Tensor, fill: str = "zero", batch: Optional[int] = None):
        """피처축 Occlusion을 통한 중요도를 계산합니다"""
        if batch is None:
            batch = agents_info.get(self.agent_id, {}).get("occlusion_batch_size", 32)
        self.eval()
        base = float(self(X_last).item())
        _, T, F = X_last.shape
        Xs = []
        for f in range(F):
            x = X_last.clone()
            if fill == "zero":
                x[:, :, f] = 0
            else:
                x[:, :, f] = X_last.mean(dim=(1, 2), keepdim=True)[:, 0, 0]
            Xs.append(x)
        deltas = []
        for i in range(0, F, batch):
            xb = torch.cat(Xs[i:i+batch], dim=0)
            yb = self(xb).flatten().cpu().numpy()
            deltas.extend(np.abs(yb - base).tolist())
        s = sum(deltas)
        return np.array([v/s if s > 0 else 1.0/F for v in deltas], dtype=float)

    def explain_last(
        self,
        X_last: torch.Tensor,
        dates: list | None = None,
        top_k: Optional[int] = None,
        use_shap: bool = True,
        shap_weight_time: Optional[float] = None,
        shap_weight_feat: Optional[float] = None
        ):
        """최신 윈도우에 대한 설명 패킷을 생성합니다"""
        cfg = agents_info.get(self.agent_id, {})
        if top_k is None:
            top_k = cfg.get("top_k_features", 5)
        if shap_weight_time is None:
            shap_weight_time = cfg.get("shap_weight_time", 0.20)
        if shap_weight_feat is None:
            shap_weight_feat = cfg.get("shap_weight_feat", 0.30)
        
        device = next(self.parameters()).device
        X_np = X_last.detach().cpu().numpy()
        X_scaled = self._scale_like_train(X_np)
        Xs = torch.tensor(X_scaled, dtype=torch.float32, device=device)

        T, F = Xs.shape[1], Xs.shape[2]
        feat_cols_src = getattr(self.stockdata, "feature_cols", [])
        feat_names = self._validate_feature_names(feat_cols_src, F)
        if dates is None:
            dates = getattr(self.stockdata, f"{self.agent_id}_dates", [])
        dates = self._validate_dates(dates, T)

        # 1. 시간 중요도 (Attention)
        time_attn = self.time_importance_from_attention(Xs)

        # 2. Grad×Input
        g_time, g_feat, gi_raw = self.gradxinput_attrib(Xs, eps=0.0)

        # 3. Occlusion
        occlusion_batch = cfg.get("occlusion_batch_size", 32)
        occ_time = self.occlusion_time(Xs, fill="zero", batch=occlusion_batch)
        occ_feat = self.occlusion_feature(Xs, fill="zero", batch=occlusion_batch)

        # 정규화
        g_time_n = g_time / (g_time.sum() + 1e-12)
        g_feat_n = g_feat / (g_feat.sum() + 1e-12)
        occ_feat_n = occ_feat / (occ_feat.sum() + 1e-12)

        # 4. SHAP (옵션)
        shap_time = None
        shap_feat = None
        shap_used = False
        if use_shap:
            try:
                shap_res = self.shap_last(Xs, background_k=64)
                shap_time = shap_res["per_time"]
                shap_feat = shap_res["per_feature"]
                shap_used = True
            except Exception:
                shap_time = None
                shap_feat = None
                shap_used = False

        attention_weights = cfg.get("attention_weights", [0.4, 0.25, 0.15])
        feature_weights = cfg.get("feature_weights", [0.5, 0.2])
        
        if shap_time is not None and shap_feat is not None:
            w_time = np.array([attention_weights[0], attention_weights[1], attention_weights[2], float(shap_weight_time)], dtype=float)
            w_time = w_time / w_time.sum()
            per_time = (
                w_time[0]*time_attn +
                w_time[1]*g_time_n +
                w_time[2]*occ_time +
                w_time[3]*shap_time
            )
            w_feat = np.array([feature_weights[0], feature_weights[1], float(shap_weight_feat)], dtype=float)
            w_feat = w_feat / w_feat.sum()
            per_feat = (
                w_feat[0]*g_feat_n +
                w_feat[1]*occ_feat_n +
                w_feat[2]*shap_feat
            )
        else:
            per_time = attention_weights[0] * time_attn + attention_weights[1] * g_time_n + attention_weights[2] * occ_time
            per_feat = feature_weights[0] * g_feat_n + feature_weights[1] * occ_feat_n

        per_time = per_time.flatten() if per_time.ndim > 1 else per_time
        per_feat = per_feat.flatten() if per_feat.ndim > 1 else per_feat

        gi_abs = np.abs(gi_raw)
        time_feature = {}
        for t_idx, d in enumerate(dates):
            pairs = sorted(
                zip(feat_names, gi_abs[t_idx].tolist()),
                key=lambda z: z[1], reverse=True
            )[:top_k]
            time_feature[str(d)] = {k: float(v) for k, v in pairs}

        time_attention = {str(d): round_num(w) for d, w in zip(dates, time_attn.tolist())}
        per_time_list = [{"date": str(d), "sum_abs": round_num(v)} for d, v in zip(dates, per_time.tolist())]
        per_feat_list = [{"feature": k, "sum_abs": round_num(v)} for k, v in sorted(zip(feat_names, per_feat.tolist()), key=lambda z: z[1], reverse=True)]

        evidence = {
            "attention": [round_num(x) for x in time_attn.tolist()],
            "gradxinput_feat": [round_num(x) for x in g_feat.tolist()],
            "occlusion_time": [round_num(x) for x in occ_time.tolist()],
            "window_size": int(T),
            "shap_used": bool(shap_used)
            }

        return {
            "per_time": per_time_list,
            "per_feature": per_feat_list,
            "time_attention": time_attention,
            "time_feature": time_feature,
            "evidence": evidence,
            "raw": {"gradxinput": gi_abs.tolist()}
          }

    def _background_windows(self, k: int = 64):
        """SHAP 계산을 위한 배경 데이터를 샘플링합니다"""
        try:
            X, _, _, _ = load_dataset(self.ticker, agent_id=self.agent_id, save_dir=self.data_dir)
            if len(X) <= 1:
                return None
            k = min(int(k), len(X) - 1)
            idx = np.linspace(0, len(X) - 2, num=k, dtype=int)
            X_bg = X[idx]
            X_bg_scaled, _ = self.scaler.transform(X_bg)
            dev = next(self.parameters()).device
            return torch.tensor(X_bg_scaled, dtype=torch.float32, device=dev)
        except Exception:
            return None

    def shap_last(self, X_last: torch.Tensor, background_k: int = 64):
        """SHAP 값을 계산합니다"""

        self.eval()
        X_bg = self._background_windows(k=background_k)
        if X_bg is None:
            X_bg = X_last.repeat(32, 1, 1)

        X_np = X_last.detach().cpu().numpy()
        X_scaled = self._scale_like_train(X_np)
        X_in = torch.tensor(X_scaled, dtype=torch.float32, device=next(self.parameters()).device)
        X_in.requires_grad_(True)

        explainer = shap.GradientExplainer(self, X_bg)
        sv = explainer.shap_values(X_in)

        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim == 2:
            sv = sv[None, ...]

        sv_abs = np.abs(sv)
        per_time = sv_abs.sum(axis=2)[0]
        per_feat = sv_abs.mean(axis=1)[0]

        per_time = per_time / (per_time.sum() + 1e-12)
        per_feat = per_feat / (per_feat.sum() + 1e-12)
        return {"per_time": per_time, "per_feature": per_feat}

    @staticmethod
    def _summarize_analysis(explanation: dict, top_time_periods: Optional[int] = None, top_features: Optional[int] = None, coverage: Optional[float] = None):
        """설명 결과를 LLM 프롬프트용으로 요약합니다"""
        agent_id = explanation.get("evidence", {}).get("agent_id", "TechnicalAgent")
        cfg = agents_info.get(agent_id, {})
        if top_time_periods is None:
            top_time_periods = cfg.get("pack_idea_top_time", 8)
        if top_features is None:
            top_features = cfg.get("pack_idea_top_feat", 6)
        if coverage is None:
            coverage = cfg.get("pack_idea_coverage", 0.8)
        
        per_time = sorted(explanation["per_time"], key=lambda z: z["sum_abs"], reverse=True)
        total = sum(z["sum_abs"] for z in per_time) or 1.0
        acc, picked = 0.0, []
        for z in per_time:
            acc += z["sum_abs"]
            picked.append({"date": z["date"], "weight": round_num(z["sum_abs"]/total)})
            if acc/total >= coverage or len(picked) >= top_time_periods:
                break

        per_feat = sorted(explanation["per_feature"], key=lambda z: z["sum_abs"], reverse=True)[:top_features]
        top_features_list = [{"feature": f["feature"], "weight": round_num(f["sum_abs"])} for f in per_feat]
        peak = picked[0]["date"] if picked else None
        return {
            "top_time": picked,
            "top_features": top_features_list,
            "peak_date": peak,
            "window_size": explanation.get("evidence",{}).get("window_size")}

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
        
        exp = self.explain_last(X_last, dates, top_k=top_k, use_shap=True)
        idea = self._summarize_analysis(exp)
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
                "uncertainty": float(my_opinion.target.uncertainty),
                "confidence": float(my_opinion.target.confidence),
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": str(target_opinion.reason)[:2000],
                "uncertainty": float(target_opinion.target.uncertainty),
                "confidence": float(target_opinion.target.confidence),
            }
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
                "confidence": float(my_opinion.target.confidence),
                "uncertainty": float(my_opinion.target.uncertainty),
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
        """데이터를 수집하고 최신 윈도우 텐서를 반환합니다"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        self.ticker = ticker
        
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        raw_csv_path = os.path.join(raw_dir, f"{ticker}_{agent_id}_raw.csv")
        
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{ticker}_{agent_id}_raw_{date_str}.csv")
            if os.path.exists(temp_path):
                raw_csv_path = temp_path
                print(f"[INFO] 백테스팅 모드: 필터링된 데이터셋 사용 ({self.simulation_date} 이전)")
        
        cfg = agents_info.get(agent_id, {})
        base_period = common_params.get("period", "2y")
        
        if base_period.endswith("y"):
            years = int(base_period[:-1])
            period_to_use = f"{years + 1}y"
        elif base_period.endswith("m"):
            months = int(base_period[:-1])
            period_to_use = f"{months + 12}m"
        else:
            period_to_use = base_period
        interval_to_use = cfg.get("interval", "1d")

        need_build = rebuild or (not os.path.exists(raw_csv_path))
        if need_build:
            is_backtest = hasattr(self, 'test_mode') and self.test_mode
            if not is_backtest or not os.path.exists(raw_csv_path):
                if not os.path.exists(raw_csv_path):
                    print(f"[{agent_id}] Raw CSV 파일이 없어 생성 중...")
                else:
                    print(f"[{agent_id}] Rebuild 요청됨. Raw CSV 재생성 중...")
                
                df = self._fetch_ticker_data(ticker, period_to_use, interval_to_use)
                feat = self._build_features_technical(df[["Open", "High", "Low", "Close", "Volume"]])
                end_date = pd.Timestamp.today().normalize()
                if base_period.endswith("y"):
                    days = int(base_period[:-1]) * 365
                elif base_period.endswith("m"):
                    days = int(base_period[:-1]) * 30
                elif base_period.endswith("d"):
                    days = int(base_period[:-1])
                else:
                    days = 2 * 365
                start_date = end_date - pd.Timedelta(days=days)
                
                try:
                    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
                    raw_tech = feat.copy()
                    raw_tech.index.name = "Date"
                    raw_tech.reset_index(inplace=True)
                    
                    raw_tech["Date"] = pd.to_datetime(raw_tech["Date"])
                    raw_tech = raw_tech[raw_tech["Date"] >= start_date].copy()
                    raw_tech = raw_tech.sort_values("Date").reset_index(drop=True)
                    
                    if "ticker" not in raw_tech.columns:
                        raw_tech.insert(1, "ticker", ticker)
                    
                    close_df = df[["Close"]].copy()
                    close_df.index.name = "Date"
                    close_df.reset_index(inplace=True)
                    close_df["Date"] = pd.to_datetime(close_df["Date"])
                    close_df = close_df[close_df["Date"] >= start_date].copy()
                    close_df = close_df.sort_values("Date").reset_index(drop=True)
                    
                    raw_tech["Date"] = raw_tech["Date"].dt.strftime("%Y-%m-%d")
                    close_df["Date"] = close_df["Date"].dt.strftime("%Y-%m-%d")
                    raw_tech = raw_tech.merge(close_df, on="Date", how="left")
                    
                    if "Close" in raw_tech.columns:
                        cols = [c for c in raw_tech.columns if c != "Close"] + ["Close"]
                        raw_tech = raw_tech[cols]
                    
                    raw_tech.to_csv(raw_csv_path, index=False)
                    print(f"✅ [{agent_id}] Raw CSV 저장 완료: {raw_csv_path} ({len(raw_tech):,} rows)")
                except Exception as e:
                    print(f"❌ [{agent_id}] Raw CSV 저장 실패: {e}")
        
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
        
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        feature_cols = cfg.get("data_cols", [])
        if not feature_cols:
            raise ValueError(f"[{agent_id}] config에 data_cols가 정의되지 않았습니다.")
        
        missing_cols = [col for col in feature_cols if col not in df_raw.columns]
        if missing_cols:
            print(f"[WARN] [{agent_id}] 누락된 feature {len(missing_cols)}개를 0.0으로 채움: {missing_cols[:5]}...")
            for col in missing_cols:
                df_raw[col] = 0.0
        
        window_size = cfg["window_size"]
        
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        x_latest = X_all[-window_size:].reshape(1, window_size, -1)
        print(f"✅ [{agent_id}] Searcher 완료: 윈도우 shape {x_latest.shape}")
        
        dates_all = df_raw["Date"].values[-window_size:].tolist()
        dates_all = [[str(d) for d in dates_all]]

        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        self.stockdata.window_size = window_size
        
        try:
            self.stockdata.last_price = float(df_raw["Close"].iloc[-1])
        except Exception:
            self.stockdata.last_price = None

        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        df_latest = pd.DataFrame(x_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)
        
        last_dates = dates_all[0] if dates_all else []
        setattr(self.stockdata, f"{agent_id}_dates_all", dates_all or [])
        setattr(self.stockdata, f"{agent_id}_dates", last_dates or [])

        return torch.tensor(x_latest, dtype=torch.float32)

    def pretrain(self):
        """TechnicalAgent 사전학습"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]
        
        if not self.ticker:
            raise ValueError("TechnicalAgent.pretrain: ticker가 설정되지 않았습니다.")
        
        ticker = self.ticker
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")
        
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        raw_csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")
        
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{ticker}_{self.agent_id}_raw_{date_str}.csv")
            if os.path.exists(temp_path):
                raw_csv_path = temp_path
                print(f"[INFO] 백테스팅 모드: 필터링된 데이터셋 사용 ({self.simulation_date} 이전)")
        
        if not os.path.exists(raw_csv_path):
            print(f"[{self.agent_id}] Raw CSV 파일이 없어 searcher() 실행 중...")
            _ = self.search(ticker, rebuild=True)
            raw_csv_path = os.path.join(raw_dir, f"{ticker}_{self.agent_id}_raw.csv")
            if not os.path.exists(raw_csv_path):
                raise FileNotFoundError(f"Raw CSV not found after searcher: {raw_csv_path}")
        
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        cfg = agents_info.get(self.agent_id, {})
        feature_cols = cfg.get("data_cols", [])
        if not feature_cols:
            raise ValueError(f"[{self.agent_id}] config에 data_cols가 정의되지 않았습니다.")
        
        missing_cols = [col for col in feature_cols if col not in df_raw.columns]
        if missing_cols:
            print(f"[WARN] [{self.agent_id}] 누락된 feature {len(missing_cols)}개를 0.0으로 채움: {missing_cols[:5]}...")
            for col in missing_cols:
                df_raw[col] = 0.0
        
        X_all = df_raw[feature_cols].values.astype(np.float32)
        close_prices = df_raw["Close"].values
        y_all = (close_prices[1:] / close_prices[:-1] - 1.0).reshape(-1, 1).astype(np.float32)
        X_all = X_all[:-1]
        
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            if len(y_all) > 0:
                y_all = y_all[:-1]
                X_all = X_all[:-1]
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중, 마지막 타겟 제거")
        
        window_size = self.window_size
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        X_seq, y_seq = self._create_sequences(X_all, y_all, window_size)
        print(f"[INFO] 시퀀스 생성 완료: {X_seq.shape}, {y_seq.shape}")
        
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_seq = y_seq * y_scale_factor
        
        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(ticker)
        
        X_train, y_train = map(torch.tensor, self.scaler.transform(X_seq, y_seq))
        X_train, y_train = X_train.float(), y_train.float()
        
        model = self
        self._modules.pop("model", None)
        
        # GPU 사용 가능 시 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        loss_fn_name = cfg.get("loss_fn", "HuberLoss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        else:
            loss_fn = torch.nn.HuberLoss()
        
        shuffle = cfg.get("shuffle", True)
        early_stopping_enabled = common_params.get("early_stopping_enabled", True)
        patience = cfg.get("patience", 20)
        min_delta = common_params.get("early_stopping_min_delta", 1e-6)
        eval_split_ratio = common_params.get("eval_split_ratio", 0.8)
        
        if early_stopping_enabled:
            split_idx = int(len(X_train) * eval_split_ratio)
            X_train_split = X_train[:split_idx]
            X_val_split = X_train[split_idx:]
            y_train_split = y_train[:split_idx]
            y_val_split = y_train[split_idx:]
            
            val_dataset = TensorDataset(X_val_split, y_val_split.view(-1, 1))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            X_train_split = X_train
            y_train_split = y_train
        
        train_dataset = TensorDataset(X_train_split, y_train_split.view(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        
        log_interval = common_params.get("pretrain_log_interval", 5)
        final_loss = None
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        
        # Early Stopping 변수 초기화
        best_val_loss_orig = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_loss_original = 0.0  # 원본 스케일 로스
            count = 0
            
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                with torch.no_grad():
                    y_pred_np = y_pred.detach().cpu().numpy()
                    yb_np = yb.detach().cpu().numpy()
                    y_pred_scaled = self.scaler.inverse_y(y_pred_np)
                    y_true_scaled = self.scaler.inverse_y(yb_np)
                    y_pred_orig = y_pred_scaled / y_scale_factor
                    y_true_orig = y_true_scaled / y_scale_factor
                    mse_orig = np.mean((y_pred_orig - y_true_orig) ** 2)
                    total_loss_original += mse_orig
                    count += 1
            
            avg_loss = total_loss / len(train_loader)
            avg_loss_original = total_loss_original / count if count > 0 else 0.0
            final_loss = avg_loss
            
            val_loss_orig = None
            if early_stopping_enabled:
                model.eval()
                val_loss_orig = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        y_pred = model(Xb)
                        y_pred_np = y_pred.cpu().numpy()
                        yb_np = yb.cpu().numpy()
                        y_pred_scaled = self.scaler.inverse_y(y_pred_np)
                        y_true_scaled = self.scaler.inverse_y(yb_np)
                        y_pred_orig = y_pred_scaled / y_scale_factor
                        y_true_orig = y_true_scaled / y_scale_factor
                        mse_orig = np.mean((y_pred_orig - y_true_orig) ** 2)
                        val_loss_orig += mse_orig
                        val_count += 1
                
                val_loss_orig /= max(val_count, 1)
                
                if val_loss_orig < (best_val_loss_orig - min_delta):
                    best_val_loss_orig = val_loss_orig
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}/{epochs} (best val loss: {best_val_loss_orig:.6f})")
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
            
            if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
                if early_stopping_enabled and val_loss_orig is not None:
                    print(f"  Epoch {epoch+1:03d}/{epochs} | Loss (scaled): {avg_loss:.6f} | Loss (original): {avg_loss_original:.6f} | Val Loss (original): {val_loss_orig:.6f}")
                else:
                    print(f"  Epoch {epoch+1:03d}/{epochs} | Loss (scaled): {avg_loss:.6f} | Loss (original): {avg_loss_original:.6f}")
        
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        self.model_loaded = True
        
        final_loss_str = f" (Final Loss: {final_loss:.6f})" if final_loss is not None else ""
        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {model_path} (device: {device}){final_loss_str}")
        
        if common_params.get("pretrain_save_dataset", True):
            dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")
            flattened_data = []
            dates_list = df_raw["Date"].values[:-1]
            
            for sample_idx in range(len(X_seq)):
                for time_idx in range(window_size):
                    date_idx = sample_idx + time_idx
                    row = {
                        'sample_id': sample_idx,
                        'time_step': time_idx,
                        'date': str(dates_list[date_idx]) if date_idx < len(dates_list) else None,
                        'target': y_seq[sample_idx, 0] if time_idx == window_size - 1 else np.nan,
                    }
                    for feat_idx, feat_name in enumerate(feature_cols):
                        row[feat_name] = X_seq[sample_idx, time_idx, feat_idx]
                    flattened_data.append(row)
            
            dataset_df = pd.DataFrame(flattened_data)
            os.makedirs(self.data_dir, exist_ok=True)
            dataset_df.to_csv(dataset_path, index=False)
            print(f"✅ 전처리된 데이터 저장 완료: {dataset_path}")

    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None, X_last: Optional[np.ndarray] = None):
        """예측 및 불확실성 추정"""
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)
        
        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다. 먼저 searcher(ticker)를 호출하세요.")
        
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        if not os.path.exists(model_path):
            if not self._in_pretrain:
                print(f"[{self.agent_id}] 모델이 없어 pretrain()을 실행합니다...")
                self._in_pretrain = True
                try:
                    self.pretrain()
                finally:
                    self._in_pretrain = False
            else:
                raise RuntimeError(f"[{self.agent_id}] pretrain 중 predict 호출로 인한 재귀 호출 방지")
        else:
            if not hasattr(self, "model_loaded") or not self.model_loaded:
                self.load_model()
        
        scaler_x_path = os.path.join(self.scaler.save_dir, f"{self.ticker}_{self.agent_id}_xscaler.pkl")
        if not os.path.exists(scaler_x_path):
            if not self._in_pretrain:
                print(f"[{self.agent_id}] 스케일러가 없어 pretrain()을 실행합니다...")
                self._in_pretrain = True
                try:
                    self.pretrain()
                finally:
                    self._in_pretrain = False
            else:
                raise RuntimeError(f"[{self.agent_id}] pretrain 중 predict 호출로 인한 재귀 호출 방지")
        
        model = self
        self.scaler.load(self.ticker)

        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                X_in = getattr(sd, self.agent_id, None)
                if isinstance(X_in, dict):
                    feature_cols = getattr(sd, "feature_cols", None)
                    if feature_cols:
                        ordered_data = {col: X_in[col] for col in feature_cols if col in X_in}
                        df = pd.DataFrame(ordered_data, columns=feature_cols)
                    else:
                        df = pd.DataFrame(X_in)
                    X_in = df.values
            if X_in is None:
                raise ValueError(f"StockData에 {self.agent_id} 데이터가 없습니다.")
            if current_price is None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            X = X_in
        
        if isinstance(X, np.ndarray):
            X_raw_np = X.copy()
        elif isinstance(X, torch.Tensor):
            X_raw_np = X.detach().cpu().numpy().copy()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        if X_raw_np.ndim == 2:
            X_raw_np = X_raw_np[None, :, :]
        elif X_raw_np.ndim == 3 and X_raw_np.shape[0] != 1:
            raise ValueError(f"예상하지 못한 배치 크기: {X_raw_np.shape[0]}")
        
        X_scaled, _ = self.scaler.transform(X_raw_np)
        device = next(model.parameters()).device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor).cpu().numpy().flatten()
                preds.append(y_pred)

        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))

        sigma = float(std_pred[-1])
        sigma_min = common_params.get("sigma_min", 1e-6)
        sigma = max(sigma, sigma_min)
        confidence = self._calculate_confidence_from_direction_accuracy()
        # 방향 정확도만 사용 (fallback 제거)

        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        if current_price is None:
            last_price = getattr(getattr(self, "stockdata", None), "last_price", None)
            default_price = common_params.get("default_current_price", 100.0)
            current_price = default_price if last_price is None else last_price

        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        predicted_return = float(mean_pred[-1]) / y_scale_factor
        
        cfg = agents_info.get(self.agent_id, {})
        return_clip_min = cfg.get("return_clip_min", -0.5)
        return_clip_max = cfg.get("return_clip_max", 0.5)
        predicted_return = np.clip(predicted_return, return_clip_min, return_clip_max)
        
        predicted_price = current_price * (1 + predicted_return)

        target = Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=confidence,
            predicted_return=float(predicted_return),
        )
        return target

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

        X, y, feature_cols, _ = load_dataset_tech(
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
