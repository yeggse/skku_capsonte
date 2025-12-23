import numpy as np
import torch
import shap
from typing import Optional, List, Dict, Union

from config.agents_set import agents_info, common_params


def round_num(value, decimals: int = 4) -> float:
    if isinstance(value, np.ndarray):
        value = float(value.mean())
    try:
        return float(f"{float(value):.{decimals}f}")
    except Exception:
        return float(value)



class TechnicalExplainer:
    """
    TechnicalAgent 전용 설명 엔진
    (Attention / Grad×Input / Occlusion / SHAP)
    """

    def __init__(self, agent):
        self.agent = agent
        self.model = agent
        self.scaler = agent.scaler

    # ------------------------------------------------------------------
    # Attention 기반 시간 중요도
    # ------------------------------------------------------------------
    @torch.no_grad()
    def time_attention(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        _ = self.model(X)
        attn = getattr(self.model, "_last_attn", None)

        T = X.shape[1]
        if attn is None:
            return np.ones(T) / T

        w = attn[0].abs().cpu().numpy()
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / T

    # ------------------------------------------------------------------
    # Grad × Input
    # ------------------------------------------------------------------
    def gradxinput(self, X: torch.Tensor):
        self.model.eval()
        x = X.clone().detach().requires_grad_(True)

        y = self.model(x).sum()
        self.model.zero_grad(set_to_none=True)
        y.backward()

        gi = (x.grad * x).abs()[0].detach().cpu().numpy()
        per_time = gi.sum(axis=1)
        per_feat = gi.mean(axis=0)
        return per_time, per_feat, gi

    # ------------------------------------------------------------------
    # Occlusion (Time)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def occlusion_time(self, X: torch.Tensor, batch: int = 32):
        base = float(self.model(X).item())
        _, T, F = X.shape

        deltas = []
        for t in range(T):
            x = X.clone()
            x[:, t, :] = 0.0
            y = float(self.model(x).item())
            deltas.append(abs(y - base))

        s = sum(deltas)
        return np.array([v / s if s > 0 else 1 / T for v in deltas])

    # ------------------------------------------------------------------
    # Occlusion (Feature)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def occlusion_feature(self, X: torch.Tensor):
        base = float(self.model(X).item())
        _, T, F = X.shape

        deltas = []
        for f in range(F):
            x = X.clone()
            x[:, :, f] = 0.0
            y = float(self.model(x).item())
            deltas.append(abs(y - base))

        s = sum(deltas)
        return np.array([v / s if s > 0 else 1 / F for v in deltas])

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def shap_last(self, X: torch.Tensor, background_k: int = 64):
        self.model.eval()
        X_bg = self._background_windows(background_k)
        if X_bg is None:
            X_bg = X.repeat(32, 1, 1)

        explainer = shap.GradientExplainer(self.model, X_bg)
        sv = explainer.shap_values(X)

        if isinstance(sv, list):
            sv = sv[0]

        sv_abs = np.abs(sv)
        per_time = sv_abs.sum(axis=2)[0]
        per_feat = sv_abs.mean(axis=1)[0]

        per_time /= (per_time.sum() + 1e-12)
        per_feat /= (per_feat.sum() + 1e-12)
        return per_time, per_feat

    def _background_windows(self, k: int):
        try:
            X, _, _, _ = self.agent.load_dataset(self.agent.ticker)
            k = min(k, len(X) - 1)
            idx = np.linspace(0, len(X) - 2, num=k, dtype=int)
            X_bg = X[idx]
            X_bg, _ = self.scaler.transform(X_bg)
            return torch.tensor(X_bg, dtype=torch.float32, device=self.agent.device)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 통합 설명 패키지
    # ------------------------------------------------------------------
    def explain(
            self,
            X_last: torch.Tensor,
            dates: List[str],
            feature_names: List[str],
            top_k: int = 5,
            use_shap: bool = True,
    ) -> Dict:
        """
        최신 윈도우(X_last)에 대한 통합 설명 생성
        - per_time: 시간별 중요도 (T,)
        - per_feature: 피처별 중요도 (F,)
        """

        # -------------------------------------------------
        # 1. 개별 설명 요소 계산
        # -------------------------------------------------
        time_attn = self.time_attention(X_last)          # (T,)
        g_time, g_feat, gi_raw = self.gradxinput(X_last) # g_time:(T,), g_feat:(F or F,k)
        occ_time = self.occlusion_time(X_last)           # (T,)
        occ_feat = self.occlusion_feature(X_last)        # (F,)

        # -------------------------------------------------
        # 2. 정규화
        # -------------------------------------------------
        time_attn = np.asarray(time_attn)
        g_time = np.asarray(g_time)
        occ_time = np.asarray(occ_time)

        g_time = g_time / (g_time.sum() + 1e-12)
        occ_time = occ_time / (occ_time.sum() + 1e-12)

        # g_feat / occ_feat → 반드시 (F,)로 축약
        g_feat = np.asarray(g_feat)
        if g_feat.ndim > 1:
            g_feat = g_feat.reshape(g_feat.shape[0], -1).mean(axis=1)

        occ_feat = np.asarray(occ_feat)
        if occ_feat.ndim > 1:
            occ_feat = occ_feat.reshape(occ_feat.shape[0], -1).mean(axis=1)

        g_feat = g_feat / (g_feat.sum() + 1e-12)
        occ_feat = occ_feat / (occ_feat.sum() + 1e-12)

        # -------------------------------------------------
        # 3. 시간 / 피처 중요도 결합
        # -------------------------------------------------
        per_time = (
                0.4 * time_attn +
                0.3 * g_time +
                0.3 * occ_time
        )

        per_feat = (
                0.6 * g_feat +
                0.4 * occ_feat
        )

        # -------------------------------------------------
        # 4. SHAP (선택)
        # -------------------------------------------------
        if use_shap:
            try:
                shap_time, shap_feat = self.shap_last(X_last)

                shap_time = np.asarray(shap_time)
                shap_feat = np.asarray(shap_feat)

                if shap_feat.ndim > 1:
                    shap_feat = shap_feat.reshape(shap_feat.shape[0], -1).mean(axis=1)

                per_time = 0.8 * per_time + 0.2 * shap_time
                per_feat = 0.7 * per_feat + 0.3 * shap_feat
            except Exception:
                # SHAP 실패 시 기존 결과 유지
                pass

        # -------------------------------------------------
        # 5. 최종 안전화 (shape 보장)
        # -------------------------------------------------
        per_time = np.asarray(per_time).reshape(-1)   # (T,)
        per_feat = np.asarray(per_feat).reshape(-1)   # (F,)

        # -------------------------------------------------
        # 6. 결과 패키징
        # -------------------------------------------------
        per_time_list = [
            {
                "date": d,
                "sum_abs": round_num(v)
            }
            for d, v in zip(dates, per_time)
        ]

        per_feat_list = sorted(
            [
                {
                    "feature": f,
                    "sum_abs": round_num(v)
                }
                for f, v in zip(feature_names, per_feat)
            ],
            key=lambda x: x["sum_abs"],
            reverse=True
        )

        return {
            "per_time": per_time_list,
            "per_feature": per_feat_list[:top_k],
            "raw": {
                "attention": time_attn.tolist(),
                "gradxinput": np.abs(gi_raw).tolist()
            }
        }



    # ------------------------------------------------------------------
    # 설명 요약 (LLM 프롬프트용)
    # ------------------------------------------------------------------
    def summarize(
            self,
            explanation: Dict,
            top_time_periods: Optional[int] = None,
            top_features: Optional[int] = None,
            coverage: Optional[float] = None,
    ) -> Dict:
        """
        explain() 결과를 LLM 프롬프트에 적합한 형태로 요약
        """

        cfg = agents_info.get(self.agent.agent_id, {})

        if top_time_periods is None:
            top_time_periods = cfg.get("pack_idea_top_time", 8)
        if top_features is None:
            top_features = cfg.get("pack_idea_top_feat", 6)
        if coverage is None:
            coverage = cfg.get("pack_idea_coverage", 0.8)

        # ---- 시간 중요도 요약 ----
        per_time = sorted(
            explanation.get("per_time", []),
            key=lambda z: z["sum_abs"],
            reverse=True
        )

        total = sum(z["sum_abs"] for z in per_time) or 1.0
        acc = 0.0
        picked_time = []

        for z in per_time:
            weight = z["sum_abs"] / total
            acc += weight
            picked_time.append({
                "date": z["date"],
                "weight": round_num(weight)
            })
            if acc >= coverage or len(picked_time) >= top_time_periods:
                break

        # ---- 피처 중요도 요약 ----
        per_feat = explanation.get("per_feature", [])[:top_features]
        picked_feat = [
            {
                "feature": f["feature"],
                "weight": round_num(f["sum_abs"])
            }
            for f in per_feat
        ]

        peak_date = picked_time[0]["date"] if picked_time else None

        return {
            "top_time": picked_time,
            "top_features": picked_feat,
            "peak_date": peak_date,
            "window_size": len(explanation.get("per_time", []))
        }
