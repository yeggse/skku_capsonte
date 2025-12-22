import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
# import shap  # Not used - removed to avoid matplotlib compatibility issues
from openai import OpenAI
from dotenv import load_dotenv

from typing import Dict, List, Optional, Literal, Tuple, Any
from collections import defaultdict
import torch
from config.agents_set import dir_info

load_dotenv()

# -----------------------------
# 데이터 구조 정의
# -----------------------------
@dataclass
class Target:
    """예측 목표값 + 불확실성 정보 포함
    - next_close: 다음 거래일 종가 예측치
    - uncertainty: Monte Carlo Dropout 기반 예측 표준편차(σ)
    - confidence: 모델 신뢰도 β (정규화된 신뢰도; 선택적)
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    feature_cols: Optional[List[str]] = None
    importances: Optional[List[float]] = None

@dataclass
class Opinion:
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str

@dataclass
class RoundLog:
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    agent_id: str = ""
    ticker: str = ""
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    feature_cols: Optional[List[str]] = None
    last_price: Optional[float] = None
    technical: Optional[Dict] = None

    def __post_init__(self):
        if self.last_price is None:
            self.last_price = 100.0


# ==============================================================
# 1️⃣ LLM 기반 설명 모듈 (확장형)
# ==============================================================
class LLMExplainer:
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model_name="gpt-4o-mini",
                 model: Optional[str] = None,
                 preferred_models: Optional[List[str]] = None,
                 temperature: float = 0.2,
                 verbose: bool = False,
                 need_training: bool = True,
                 ):

        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
        if not self.api_key:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name

        self.agent_id = 'MacroAgent'
        self.temperature = temperature # Temperature 설정
        self.verbose = verbose            # 디버깅 모드
        self.need_training = need_training # 모델 학습 필요 여부
        # 모델 폴백 우선순위
        self.preferred_models = preferred_models or ["gpt-5-mini", "gpt-4.1-mini"]
        if model:
            self.preferred_models = [model] + [
                m for m in self.preferred_models if m != model
            ]

        # 공통 헤더
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 상태값
        self.stockdata: Optional[StockData] = None
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)

        # JSON Schema
        self.schema_obj_opinion = {
            "type": "object",
            "properties": {
                "next_close": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["next_close", "reason"],
            "additionalProperties": False,
        }
        self.schema_obj_rebuttal = {
            "type": "object",
            "properties": {
                "stance": {"type": "string", "enum": ["REBUT", "SUPPORT"]},
                "message": {"type": "string"},
            },
            "required": ["stance", "message"],
            "additionalProperties": False,
        }

    def generate_explanation(
            self,
            feature_summary,
            predictions,
            importance_summary,
            temporal_summary=None,
            consistency_summary=None,
            sensitivity_summary=None,
            stability_summary=None,
            stock_data=None,
            target=None,
    ):
        """
        Gradient × Input / Integrated Gradients 기반 feature importance 결과를 바탕으로
        LLM이 논리적 금융 분석을 생성하도록 하는 버전
        """

        def _summarize(obj, max_len=1500):
            text = str(obj)
            if len(text) > max_len:
                text = text[:max_len] + "\n...(truncated)"
            return text

        # ✅ 안전한 문자열 변환
        importance_summary = _summarize(importance_summary)
        temporal_summary = _summarize(temporal_summary)
        consistency_summary = _summarize(consistency_summary)
        sensitivity_summary = _summarize(sensitivity_summary)
        stability_summary = _summarize(stability_summary)

        # 1️⃣ system 메시지
        sys_text = (
            "너는 금융 시장을 분석하는 인공지능 애널리스트이다. "
            "Gradient × Input 및 Integrated Gradients 기반의 LSTM 예측 결과를 해석해야 한다. "
            "모델의 예측값, 변수 중요도, 시간적 변화, 일관성, 민감도, 안정성을 종합적으로 고려하여 "
            "경제적 의미를 도출하라."
        )

        # 2️⃣ user 메시지 (Gradient 기반 분석 중심)
        user_text = f"""
        ### 1. 모델 예측 결과
        {predictions}
    
        ### 2. 주요 변수 중요도 요약 (feature_summary)
        {feature_summary}
    
        ### 3. 전체 변수 중요도 맵 (importance_dict)
        {importance_summary}
    
        ### 4. 상위 변수 및 시점별 영향 변화 (temporal_summary)
        {temporal_summary}
    
        ### 5. IG / G×I 간 일관성 분석 (consistency_summary)
        {consistency_summary}
    
        ### 6. 입력 변화 민감도 분석 (sensitivity_summary)
        {sensitivity_summary}
    
        ### 7. 변수 중요도 안정성 분석 (stability_summary)
        {stability_summary}
    
        ---  
        위 데이터를 바탕으로 아래 항목을 중심으로 체계적이고 논리적으로 분석하세요.
        
        (1) **Feature Trend (Temporal) 분석:**
            - 어떤 변수들의 영향력이 최근 시점으로 갈수록 커졌습니까?
            - 반대로 영향력이 약화된 변수는 무엇입니까?
            - 이러한 변화가 나타난 거시적·산업적 요인은 무엇입니까?
            - 시간 흐름에 따른 변수 영향 변화가 예측 방향에 어떤 의미를 가지는지 설명하십시오.
        
        (2) **Model Consistency 분석:**
            - Integrated Gradients와 Gradient × Input 결과가 일치하는 주요 feature와 불일치하는 feature를 구분하십시오.
            - 불일치가 높은 feature는 어떤 시장 불확실성, 데이터 잡음, 또는 비선형 구조에 의해 발생했을 가능성이 있습니까?
            - 일관성이 높은 변수군이 모델이 신뢰할 만한 구조적 요인을 반영하고 있는지 논의하십시오.
        
        (3) **Sensitivity (민감도) 분석:**
            - 입력값의 작은 변화에 큰 예측 변화가 발생한 변수는 무엇입니까?
            - 민감도가 높다는 것은 해당 feature가 단기 시장 변동성 또는 과민 반응에 민감함을 의미합니다. 
              이러한 feature들이 포트폴리오 리스크나 단기 트레이딩 전략에 어떤 시사점을 주는지 분석하십시오.
            - 민감도가 낮은 feature는 어떤 안정적 요인을 반영하는지 설명하십시오.
        
        (4) **Stability (안정성) 분석:**
            - 학습 구간이나 샘플링 변화에 따라 feature 중요도의 변동 폭이 큰 변수는 무엇입니까?
            - 변동성이 높은 변수는 시장 국면 전환이나 뉴스 이벤트에 반응할 가능성이 있습니다.
            - 반대로 변동성이 낮은 변수들은 구조적·장기적 트렌드에 연동된 요인일 수 있습니다. 
              이러한 차이를 금융적으로 해석하십시오.
        
        (5) **통합 결론 (Integrated Insight):**
            - 위 네 가지 관점을 종합하여 이번 예측의 주요 원동력을 설명하십시오.
            - 어떤 변수 조합이 향후 가격 움직임에 가장 큰 영향을 미칠 것으로 예상되는지 논리적으로 제시하십시오.
            - 모델의 신뢰성과 해석 가능성을 동시에 고려하여, 예측 결과에 대한 전문가적 평가를 작성하십시오.

        
        ---
        추가 맥락:
        최근 종가: {getattr(stock_data, 'last_price', 'N/A')}
        예측 종가: {getattr(target, 'next_close', 'N/A')}
        """

        # 3️⃣ 메시지 빌드 (system + user)
        msg_sys = self._msg("system", sys_text)
        msg_user = self._msg("user", user_text)

        # 4️⃣ LLM 호출
        parsed = self._ask_with_fallback(msg_sys, msg_user, self.schema_obj_opinion)
        reason = parsed.get("reason") or "(사유 생성 실패: 미입력)"

        return reason



    #[base_agent.py]
    def _msg(self, role: str, content: str) -> dict:
        """OpenAI ChatCompletion용 메시지 구조 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}


    #[base_agent.py] OpenAI API 호출
    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """Chat Completions API 호출 (fallback 지원)"""
        last_err = None
        for model in self.preferred_models:
            payload = {
                "model": model,
                "messages": [msg_sys, msg_user],
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Response",
                        "schema": schema_obj
                    }
                }
            }
            try:
                import requests
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    # 최신 Chat API의 응답 처리
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not msg:
                        continue
                    try:
                        return json.loads(msg)
                    except Exception:
                        return {"reason": msg.strip()}
                else:
                    last_err = (r.status_code, r.text)
                    continue
            except Exception as e:
                last_err = str(e)
                continue
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    def _p(self, msg: str):
        if self.verbose:
            print(f"[{self.agent_id}] {msg}")






# ==============================================================
# GradientAnalyzer (Integrated Gradients (IG) 와 Gradient × Input (G×I))
# ==============================================================
class GradientAnalyzer:
    """
    Gradient × Input + Integrated Gradients 기반 피처 중요도 분석기
    - SHAP을 대체하며, LSTM 등 시계열 모델에도 안정적으로 동작
    - 두 방법 간 상관계수를 통해 일관성 검증 및 중요도 통합
    """

    def __init__(self, model, feature_names, baseline=None, steps:int=50):
        self.model = model
        self.feature_names = feature_names
        self.baseline = baseline
        self.steps = steps

    # ------------------------------------------------------------
    # 1️⃣ Gradient × Input 계산
    # ------------------------------------------------------------
    def compute_gradient_x_input(self, x_input: np.ndarray) -> np.ndarray:
        """
        Gradient × Input 계산 (PyTorch)
        - 입력 차원을 (batch, time, features) 형태로 강제 정규화
        - (1, 1, 40, 169) 같은 잘못된 입력도 자동 수정
        """
        # ✅ 차원 정규화
        x_input = np.array(x_input, dtype=np.float32)
        if x_input.ndim == 4:
            # (1, 1, 40, features) -> (1, 40, features)
            x_input = np.squeeze(x_input, axis=1)
        elif x_input.ndim == 2:
            # (40, features) -> (1, 40, features)
            x_input = np.expand_dims(x_input, axis=0)

        # ✅ PyTorch Tensor 변환 및 Gradient 계산
        device = next(self.model.parameters()).device
        x = torch.FloatTensor(x_input).to(device)
        x.requires_grad_(True)
        
        self.model.eval()
        preds = self.model(x)
        
        # Gradient 계산
        grads = torch.autograd.grad(
            outputs=preds.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False
        )[0]
        
        gx = torch.abs(grads * x)

        return gx.detach().cpu().numpy()


    # ------------------------------------------------------------
    # 2️⃣ Integrated Gradients 계산
    # ------------------------------------------------------------
    def compute_integrated_gradients(self, x_input: np.ndarray) -> np.ndarray:
        # ✅ 차원 정리: (batch, time, features)
        x_input = np.array(x_input, dtype=np.float32)
        if x_input.ndim == 4:
            # (steps, 1, 40, features) or (1, 1, 40, features)
            x_input = np.squeeze(x_input, axis=1)
        if x_input.ndim == 2:
            # (40, features) -> (1, 40, features)
            x_input = np.expand_dims(x_input, axis=0)

        if self.baseline is None:
            self.baseline = np.zeros_like(x_input)

        # ✅ baseline과 shape 동일 확인
        assert self.baseline.shape == x_input.shape, \
            f"Baseline shape {self.baseline.shape} != x_input {x_input.shape}"

        interpolated = [
            self.baseline + (float(i)/self.steps)*(x_input - self.baseline)
            for i in range(self.steps + 1)
        ]
        interpolated = np.array(interpolated, dtype=np.float32)  # (steps+1, 1, 40, features) 형태
        interpolated = np.squeeze(interpolated, axis=1)          # ✅ (steps+1, 40, features)

        # ✅ PyTorch Tensor 변환 및 Gradient 계산
        device = next(self.model.parameters()).device
        interpolated_torch = torch.FloatTensor(interpolated).to(device)
        interpolated_torch.requires_grad_(True)
        
        self.model.eval()
        preds = self.model(interpolated_torch)

        # Gradient 계산
        grads = torch.autograd.grad(
            outputs=preds.sum(),
            inputs=interpolated_torch,
            create_graph=False,
            retain_graph=False
        )[0]
        
        avg_grads = torch.mean(grads[:-1], dim=0)
        ig = (x_input - self.baseline) * avg_grads.detach().cpu().numpy()

        return ig

    # ------------------------------------------------------------
    # 3️⃣ 병합 실행 (SHAP 대체)
    # ------------------------------------------------------------
    def run_all_gradients(self, x_input: np.ndarray):
        """
        Gradient × Input + Integrated Gradients를 동시에 수행하고
        6가지 summary 구조로 feature importance를 반환하는 버전.
        """

        # Gradient analysis 실행 중...

        # 1️⃣ Gradient × Input / Integrated Gradients 계산
        gx = self.compute_gradient_x_input(x_input)
        ig = self.compute_integrated_gradients(x_input)

        gx_mean = np.mean(np.abs(gx), axis=(0, 1))
        ig_mean = np.mean(np.abs(ig), axis=(0, 1))

        feature_names = np.array(self.feature_names)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "gradxinput": gx_mean,
            "integrated_gradients": ig_mean
        })

        # 2️⃣ 두 attribution의 평균을 최종 중요도로 사용
        importance_df["final_importance"] = (
                0.5 * (importance_df["gradxinput"] + importance_df["integrated_gradients"])
        )

        # 3️⃣ 일관성(agreement ratio)
        corr = np.corrcoef(gx_mean, ig_mean)[0, 1]
        # IG–G×I correlation: {corr:.4f}

        # 4️⃣ feature summary (핵심 요약)
        feature_summary = {
            "agreement_ratio": float(corr),
            "gx_importance_top": importance_df.sort_values("gradxinput", ascending=False).head(3)["feature"].tolist(),
            "ig_importance_top": importance_df.sort_values("integrated_gradients", ascending=False).head(3)["feature"].tolist()
        }

        # 5️⃣ importance dict
        importance_dict = dict(
            zip(feature_names, importance_df["final_importance"])
        )

        # 6️⃣ temporal summary (상위 5개 feature 세부요약)
        temporal_summary = (
            importance_df.sort_values("final_importance", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )

        # 7️⃣ consistency summary (IG vs G×I 순위 일치도)
        ig_rank = importance_df.sort_values("integrated_gradients", ascending=False).reset_index(drop=True)
        gx_rank = importance_df.sort_values("gradxinput", ascending=False).reset_index(drop=True)
        consistency_summary = []
        for f in feature_names:
            ig_pos = ig_rank[ig_rank["feature"] == f].index[0]
            gx_pos = gx_rank[gx_rank["feature"] == f].index[0]
            rank_gap = abs(int(ig_pos) - int(gx_pos))
            if rank_gap > 10:  # 순위 차이가 큰 feature만 저장
                consistency_summary.append({"feature": f, "rank_gap": rank_gap})

        # 8️⃣ sensitivity summary (gradient 표준편차 기반 민감도)
        grads = np.abs(gx)
        sensitivity_summary = [
            {"feature": f, "sensitivity": float(np.std(grads[:, :, i]))}
            for i, f in enumerate(feature_names)
        ]
        sensitivity_summary = sorted(sensitivity_summary, key=lambda x: x["sensitivity"], reverse=True)[:5]

        # 9️⃣ stability summary (feature 중요도의 변동성)
        importance_df["variance"] = importance_df[["gradxinput", "integrated_gradients"]].var(axis=1)
        stability_summary = (
            importance_df.sort_values("variance", ascending=False)
            .head(5)
            .to_dict(orient="records")
        )


        # 🔟 모든 summary 통합
        grad_results = {
            "feature_summary": feature_summary,
            "importance_dict": importance_dict,
            "temporal_summary": temporal_summary,
            "consistency_summary": consistency_summary,
            "sensitivity_summary": sensitivity_summary,
            "stability_summary": stability_summary
        }

        # Gradient analysis 완료
        return (importance_dict, pd.DataFrame(temporal_summary), pd.DataFrame(consistency_summary),
                pd.DataFrame(sensitivity_summary), grad_results)
