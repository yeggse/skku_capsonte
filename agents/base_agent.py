# ===============================================================
# BaseAgent: LLM 기반 공통 인터페이스
# ===============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple, Any
from dataclasses import field
from collections import defaultdict
import os, json, requests
from datetime import datetime
from dotenv import load_dotenv
from config.agents_set import agents_info, dir_info, common_params
from core.data_set import build_dataset, load_dataset
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import torch
import pandas as pd
import yfinance as yf

# ===============================================================
# 데이터 구조 정의
# ===============================================================
@dataclass
class Target:
    """
    예측 목표값 및 불확실성 정보를 담는 데이터 클래스
    
    Attributes:
        next_close: 다음 거래일의 예측 종가
        uncertainty: 예측의 불확실성
        confidence: 모델의 신뢰도 (0~1)
        predicted_return: 예측 수익률
    """
    next_close: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    predicted_return: Optional[float] = None

@dataclass
class Opinion:
    """
    에이전트의 예측 의견을 담는 클래스
    
    Attributes:
        agent_id: 의견을 제시한 에이전트 ID
        target: 예측 가격 및 불확실성 정보
        reason: 예측의 근거
    """
    agent_id: str
    target: Target
    reason: str

@dataclass
class Rebuttal:
    """
    타 에이전트 의견에 대한 반박/지지 메시지
    
    Attributes:
        from_agent_id: 발신 에이전트 ID
        to_agent_id: 수신 에이전트 ID
        stance: 반박(REBUT) 또는 지지(SUPPORT) 입장
        message: 반박 또는 지지의 상세 내용
        support_rate: 지지율 (0~1, SUPPORT일 때만 유효, REBUT일 때는 0)
    """
    from_agent_id: str
    to_agent_id: str
    stance: Literal["REBUT", "SUPPORT"]
    message: str
    support_rate: Optional[float] = None

@dataclass
class RoundLog:
    """
    토론 라운드별 로그 데이터
    
    Attributes:
        round_no (int): 라운드 번호
        opinions (List[Opinion]): 해당 라운드의 에이전트별 의견 목록
        rebuttals (List[Rebuttal]): 해당 라운드의 반박 메시지 목록
        summary (Dict[str, Target]): 라운드 요약 정보
    """
    round_no: int
    opinions: List[Opinion]
    rebuttals: List[Rebuttal]
    summary: Dict[str, Target]

@dataclass
class StockData:
    """
    에이전트가 사용하는 주식 데이터 컨테이너
    
    Attributes:
        SentimentalAgent (Optional[Dict]): 감성 분석 데이터
        MacroAgent (Optional[Dict]): 거시경제 데이터
        TechnicalAgent (Optional[Dict]): 기술적 분석 데이터
        last_price (Optional[float]): 최신 종가
        currency (Optional[str]): 통화 코드 (예: USD)
        ticker (Optional[str]): 종목 코드
        feature_cols (Optional[List[str]]): 피처 컬럼 이름 목록
    """
    SentimentalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    MacroAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    TechnicalAgent: Optional[Dict[str, Any]] = field(default_factory=dict)
    last_price: Optional[float] = None
    currency: Optional[str] = None
    ticker: Optional[str] = None
    feature_cols: Optional[List[str]] = field(default_factory=list)


# ===============================================================
# BaseAgent 클래스
# ===============================================================
class BaseAgent:
    """
    LLM 기반 Multi-Agent Debate 시스템을 위한 기본 에이전트 클래스.
    모든 개별 에이전트는 이 클래스를 상속받아야 합니다.
    
    주요 기능:
    - 데이터 로드 및 전처리
    - 모델 학습 및 관리
    - 예측 및 불확실성 추정
    - LLM 기반 의견 생성 및 토론 참여
    """

    OPENAI_URL = "https://api.openai.com/v1/responses"

    def __init__(
            self,
            agent_id: str,
            model: Optional[str] = None,
            preferred_models: Optional[List[str]] = None,
            temperature: Optional[float] = None,
            verbose: bool = False,
            need_training: bool = True,
            data_dir: str = dir_info["data_dir"],
            model_dir: str = dir_info["model_dir"],
            ticker: str=None,
            gamma: Optional[float] = None,
            delta_limit: Optional[float] = None,
    ):
        """
        BaseAgent 초기화
        
        Args:
            agent_id: 에이전트 식별자
            model: 사용할 LLM 모델명
            preferred_models: 모델 폴백 우선순위 리스트
            temperature: LLM 생성 온도
            verbose: 디버그 출력 여부
            need_training: 학습 필요 여부
            data_dir: 데이터 저장 경로
            model_dir: 모델 저장 경로
            ticker: 종목 코드
            gamma: 의견 수렴율 (0~1)
            delta_limit: 최대 변화 허용 폭
        """
        load_dotenv()
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature if temperature is not None else common_params.get("temperature", 0.2)
        self.verbose = verbose
        self.need_training = need_training
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.ticker = ticker
        
        scaler_dir = os.path.join(model_dir, "scalers")
        self.scaler = DataScaler(agent_id, scaler_dir=scaler_dir)
        self.window_size = agents_info[agent_id]["window_size"]
        self.preferred_models = preferred_models or common_params.get("preferred_models", ["gpt-5-mini", "gpt-4.1-mini"])
        self._in_pretrain = False
        self._calculating_confidence = False
        
        if model:
            self.preferred_models = [model] + [m for m in self.preferred_models if m != model]

        self.api_key = os.getenv("CAPSTONE_OPENAI_API")
        if not self.api_key:
            raise RuntimeError("환경변수 CAPSTONE_OPENAI_API가 설정되지 않았습니다.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.stockdata: Optional[StockData] = None
        self.targets: List[Target] = []
        self.opinions: List[Opinion] = []
        self.rebuttals: Dict[int, List[Rebuttal]] = defaultdict(list)
        self.gamma = gamma if gamma is not None else agents_info[agent_id].get("gamma", 0.3)
        self.delta_limit = delta_limit if delta_limit is not None else agents_info[agent_id].get("delta_limit", 0.05)

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
                "support_rate": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "지지율 (0~1). SUPPORT일 때만 유효, REBUT일 때는 0"
                }
            },
            "required": ["stance", "message", "support_rate"],
            "additionalProperties": False,
        }

    def set_test_mode(self, mode: bool):
        """테스트 모드 활성화/비활성화"""
        self.test_mode = mode

    def set_simulation_date(self, date_str: str):
        """시뮬레이션 기준 날짜 설정"""
        self.simulation_date = date_str
        self.test_mode = True

    def set_training_window(self, start_date: str):
        """학습 시작 날짜 설정"""
        self.training_start_date = start_date

    def search(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        데이터를 검색하고 준비하는 메서드
        
        Args:
            ticker: 종목 코드
            rebuild: 데이터셋 강제 재생성 여부
            
        Returns:
            torch.Tensor: 모델 입력용 텐서
        """


        agent_id = self.agent_id

        if ticker is None:
            ticker = self.ticker

        self.ticker = ticker

        dataset_path = os.path.join(self.data_dir, f"{ticker}_{agent_id}_dataset.csv")

        if not os.path.exists(dataset_path) or rebuild:
            print(f"⚙️ {ticker} {agent_id} dataset not found. Building new dataset...")
            build_dataset(ticker=ticker, save_dir=self.data_dir)

        X, y, feature_cols = load_dataset(ticker, agent_id=agent_id, save_dir=self.data_dir)
        self.stockdata = StockData()
        x_latest = X[-1:]
        X_tensor = torch.tensor(x_latest, dtype=torch.float32)
        df_latest = pd.DataFrame(x_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)
        self.stockdata.ticker = ticker
        try:
            data = yf.download(ticker, period="1d", interval="1d", auto_adjust=False, progress=False)
            val = data["Close"].iloc[-1]
            self.stockdata.last_price = float(val.item() if hasattr(val, "item") else val)
        except Exception as e:
            print(f"yfinance 오류 발생 (last_price)")

        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception as e:
            print(f"yfinance 오류 발생, 통화 기본값 사용: {e}")
            self.stockdata.currency = "USD"

        return X_tensor

    def _extract_current_price(self, input_data, input_array=None, current_price=None) -> float:
        """현재가를 추출하는 내부 헬퍼 메서드"""
        if current_price is not None:
            return float(current_price)

        sd = None

        if isinstance(input_data, StockData):
            sd = input_data
        elif hasattr(self, "stockdata"):
            sd = getattr(self, "stockdata", None)

        if sd is not None:
            snap = getattr(sd, "snapshot", None) or getattr(sd, "meta", None) or {}
            if isinstance(snap, dict):
                for key in ("last_price", "current_price", "close", "adj_close"):
                    v = snap.get(key)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
            if getattr(sd, "last_price", None) is not None:
                return float(sd.last_price)

        if input_array is not None and np.ndim(input_array) >= 2:
            last_step = input_array[-1]
            if np.ndim(last_step) == 2:
                last_step = last_step[-1]
            try:
                return float(last_step[-1])
            except Exception:
                pass

        raise RuntimeError(
            "[BaseAgent.predict] current_price를 찾을 수 없습니다. "
            "current_price를 전달하거나 StockData에 last_price를 설정하세요."
        )

    def _calculate_confidence_from_direction_accuracy(self) -> Optional[float]:
        """
        최근 N일 동안의 방향정확도를 계산하여 신뢰도로 반환
        
        Returns:
            float: 방향정확도 기반 신뢰도 (0~1 범위), 계산 실패시 None
        """

        try:
            lookback_days = common_params.get("confidence_lookback_days", 30)
            
            if not self.ticker or not self.agent_id or not self.data_dir:
                return None
            
            dataset_path = os.path.join(self.data_dir, f"{self.ticker}_{self.agent_id}_dataset.csv")
            if not os.path.exists(dataset_path):
                return None
            
            df = pd.read_csv(dataset_path)
            meta_cols = {"sample_id", "time_step", "target", "date"}
            feature_cols = [
                c for c in df.columns
                if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
            
            if len(feature_cols) == 0:
                return None
            
            unique_samples = sorted(df['sample_id'].unique())
            if len(unique_samples) < lookback_days:
                return None
            
            recent_samples = unique_samples[-lookback_days:]
            
            # TechnicalAgent, MacroAgent는 nn.Module을 상속받아 self가 모델
            if self.model is None:
                if hasattr(self, 'forward') and isinstance(self, torch.nn.Module):
                    model_to_use = self
                else:
                    return None
            else:
                model_to_use = self.model
            
            correct_count = 0
            total_count = 0
            
            if self._calculating_confidence:
                return None
            
            self._calculating_confidence = True
            
            try:
                if hasattr(self, "device"):
                    device = self.device
                elif hasattr(model_to_use, "parameters"):
                    try:
                        device = next(model_to_use.parameters()).device
                    except StopIteration:
                        device = torch.device("cpu")
                else:
                    device = torch.device("cpu")
                
                model_to_use.eval()
                
                for sample_id in recent_samples:
                    try:
                        sample_data = df[df['sample_id'] == sample_id].sort_values('time_step')
                        if len(sample_data) == 0:
                            continue
                        
                        X_sample = sample_data[feature_cols].values.astype(np.float32)
                        y_actual = sample_data['target'].iloc[-1]
                        
                        if np.isnan(y_actual) or np.any(np.isnan(X_sample)):
                            continue
                        
                        # y_actual도 역변환 (스케일링된 값을 원본 수익률로)
                        if hasattr(self, "scaler") and hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
                            try:
                                y_actual_scaled = np.array([[y_actual]])
                                y_actual_inverse = self.scaler.inverse_y(y_actual_scaled)
                                # inverse_y가 1차원 배열을 반환하는 경우 처리
                                if isinstance(y_actual_inverse, np.ndarray):
                                    if y_actual_inverse.ndim == 1:
                                        y_actual = y_actual_inverse[0]
                                    else:
                                        y_actual = y_actual_inverse[0, 0]
                                else:
                                    y_actual = y_actual_inverse
                            except Exception:
                                pass
                        
                        X_tensor = torch.from_numpy(X_sample).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            out = model_to_use(X_tensor)
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            y_pred = out.detach().cpu().numpy().squeeze()
                        
                        if hasattr(self, "scaler") and hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
                            try:
                                y_pred_scaled = np.array([[y_pred]])
                                y_pred = self.scaler.inverse_y(y_pred_scaled)[0, 0]
                            except Exception:
                                pass
                        
                        # 이제 둘 다 원본 수익률로 비교
                        if np.sign(y_pred) == np.sign(y_actual):
                            correct_count += 1
                        total_count += 1
                        
                    except Exception as e:
                        continue
                
            finally:
                self._calculating_confidence = False
            
            if total_count == 0:
                return None
            
            direction_accuracy = correct_count / total_count
            return float(direction_accuracy)
            
        except Exception as e:
            return None


    def review_draft(self, stock_data=None, target=None):
        """초기 의견을 생성합니다"""
        if stock_data is None:
            sd = getattr(self, "stockdata", None)
            if sd is None:
                try:
                    self.search()
                    sd = self.stockdata
                except Exception:
                    pass
            
            if sd is None:
                raise RuntimeError(f"[{self.agent_id}] StockData가 없습니다.")
            
            if isinstance(sd, dict):
                stock_data = sd.get(self.agent_id, None)
            else:
                stock_data = sd

        if target is None:
            target = self.predict(stock_data)

        sys_text, user_text = self._build_messages_opinion(self.stockdata, target)

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )

        reason = parsed.get("reason", "(사유 생성 실패)")
        self.opinions.append(Opinion(agent_id=self.agent_id, target=target, reason=reason))
        return self.opinions[-1]

    def review_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """상대방 의견에 대한 반박을 생성합니다"""
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
            print(f"[{self.agent_id}] Rebuttal: {result.stance} -> {other_opinion.agent_id}, support_rate: {support_rate}")

        return result

    def _calculate_consensus_price(
        self, 
        my_opinion: Opinion, 
        others: List[Opinion],
        rebuttals: Optional[List[Rebuttal]] = None
    ) -> float:
        """
        불확실성, 신뢰도, SUPPORT RATE를 모두 반영하여 합의된 가격을 계산합니다.
        
        각 에이전트의 support_rate를 gamma(수용률)로 사용하여:
        - REBUT(support_rate=0)은 자동 배제
        - SUPPORT는 support_rate만큼만 반영
        """
        try:
            my_price = float(my_opinion.target.next_close)
            sigma_min = common_params.get("sigma_min", 1e-6)
            my_sigma = abs(my_opinion.target.uncertainty or sigma_min)
            my_confidence = my_opinion.target.confidence or 0.5  # 기본값 0.5

            if not others:
                return my_price

            # rebuttals를 딕셔너리로 변환 (from_agent_id -> support_rate)
            support_rates = {}
            if rebuttals:
                for rebuttal in rebuttals:
                    from_agent = rebuttal.from_agent_id
                    support_rate = rebuttal.support_rate
                    if support_rate is None:
                        support_rate = 0.0 if rebuttal.stance == "REBUT" else 0.5
                    support_rates[from_agent] = support_rate

            other_prices = np.array([o.target.next_close for o in others], dtype=float)
            other_sigmas = np.array([abs(o.target.uncertainty or sigma_min) for o in others], dtype=float)
            other_confidences = np.array([o.target.confidence or 0.5 for o in others], dtype=float)
            
            # 1. 불확실성 기반 가중치 계산
            all_sigmas = np.concatenate([[my_sigma], other_sigmas])
            inv_sigmas = 1 / (all_sigmas + sigma_min)
            betas_uncertainty = inv_sigmas / inv_sigmas.sum()
            betas_others_uncertainty = betas_uncertainty[1:]
            
            # 2. 신뢰도 기반 가중치 계산
            all_confidences = np.concatenate([[my_confidence], other_confidences])
            betas_confidence = all_confidences / (all_confidences.sum() + 1e-10)
            betas_others_confidence = betas_confidence[1:]
            
            # 3. 불확실성 × 신뢰도 결합 가중치
            combined_reliability = betas_others_uncertainty * betas_others_confidence
            if combined_reliability.sum() > 0:
                combined_reliability = combined_reliability / combined_reliability.sum()
            else:
                # 모든 가중치가 0이면 불확실성 가중치만 사용
                combined_reliability = betas_others_uncertainty
            
            # 4. 각 에이전트별로 support_rate를 gamma로 사용하여 delta 계산
            delta = 0.0
            for i, other_opinion in enumerate(others):
                other_agent_id = other_opinion.agent_id
                other_price = other_prices[i]
                beta_i = combined_reliability[i]
                
                # 해당 에이전트의 support_rate를 gamma로 사용
                agent_gamma = support_rates.get(other_agent_id, 0.5)  # 기본값 0.5
                
                # 각 에이전트별 delta 계산: support_rate × 가중치 × 가격차이
                agent_delta = agent_gamma * beta_i * (other_price - my_price)
                delta += agent_delta
            
            # 5. 최종 가격 (gamma 없이 바로 적용)
            revised_price = my_price + delta
            return float(revised_price)

        except Exception as e:
            print(f"[{self.agent_id}] _calculate_consensus_price 실패: {e}")
            return float(my_opinion.target.next_close)

    def review_revise(
            self,
            my_opinion: Opinion,
            others: List[Opinion],
            rebuttals: List[Rebuttal],
            stock_data: StockData,
            fine_tune: bool = True,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
    ):
        """토론 후 자신의 예측을 수정합니다"""
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", common_params.get("fine_tune_epochs", 10))
        
        revised_price = self._calculate_consensus_price(my_opinion, others, rebuttals)
        x_latest = None
        loss_value = None
        model = getattr(self, "model", None)
        if model is None and isinstance(self, torch.nn.Module):
            model = self

        try:
            x_latest = self.search(self.ticker)
        except Exception as e:
            print(f"[{self.agent_id}] searcher 호출 실패: {e}")
            predicted_target = Target(
                next_close=float(revised_price),
                uncertainty=my_opinion.target.uncertainty,
                confidence=my_opinion.target.confidence
            )
            try:
                sys_text, user_text = self._build_messages_revision(
                    my_opinion=my_opinion,
                    others=others,
                    rebuttals=rebuttals,
                    stock_data=stock_data,
                )
            except Exception as e:
                print(f"[{self.agent_id}] Revision 메시지 생성 실패: {e}")
                sys_text, user_text = ("금융 분석가입니다.", json.dumps({"reason": "메시지 생성 실패"}))
            
            parsed = self._ask_with_fallback(
                self._msg("system", sys_text),
                self._msg("user", user_text),
                {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            )
            revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
            revised_opinion = Opinion(
                agent_id=self.agent_id,
                target=predicted_target,
                reason=revised_reason,
            )
            self.opinions.append(revised_opinion)
            return revised_opinion

        if fine_tune and model is not None and x_latest is not None:
            try:
                current_price = getattr(stock_data, "last_price", None)
                if current_price is None:
                    default_price = common_params.get("default_current_price", 100.0)
                    current_price = getattr(self, "last_price", default_price)

                revised_return = (revised_price / current_price) - 1.0
                y_scale_factor = common_params.get("y_scale_factor", 100.0)
                revised_return_scaled = revised_return * y_scale_factor
                
                if hasattr(self, "scaler") and getattr(self.scaler, "y_scaler", None) is not None:
                    y_target_scaled = self.scaler.y_scaler.transform(
                        np.array([[revised_return_scaled]], dtype=float)
                    )[0, 0]
                else:
                    y_target_scaled = revised_return_scaled

                device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
                
                if isinstance(x_latest, torch.Tensor):
                    X_tensor = x_latest.to(device).float()
                else:
                    X_tensor = torch.tensor(x_latest, dtype=torch.float32).to(device)
                
                y_tensor = torch.tensor([[y_target_scaled]], dtype=torch.float32).to(device)

                model.train()
                try:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    huber_delta = common_params.get("huber_loss_delta", 1.0)
                    criterion = torch.nn.HuberLoss(delta=huber_delta)

                    for _ in range(epochs):
                        optimizer.zero_grad()
                        pred = model(X_tensor)
                        loss = criterion(pred, y_tensor)
                        loss.backward()
                        optimizer.step()

                    loss_value = float(loss.item())
                    print(f"[{self.agent_id}] Fine-tuning 완료: loss={loss_value:.6f}")
                finally:
                    model.eval()

            except Exception as e:
                print(f"[{self.agent_id}] Fine-tuning 실패: {e}")

        try:
            predicted_target = self.predict(x_latest, current_price=getattr(stock_data, "last_price", None))
        except Exception as e:
            print(f"[{self.agent_id}] 재예측 실패, 합의 가격 사용: {e}")
            predicted_target = Target(
                next_close=float(revised_price),
                uncertainty=my_opinion.target.uncertainty,
                confidence=my_opinion.target.confidence
            )

        try:
            sys_text, user_text = self._build_messages_revision(
                my_opinion=my_opinion,
                others=others,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )
        except Exception as e:
            print(f"[{self.agent_id}] Revision 메시지 생성 실패: {e}")
            sys_text, user_text = ("금융 분석가입니다.", json.dumps({"reason": "메시지 생성 실패"}))

        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        )

        revised_reason = parsed.get("reason", "(수정 사유 생성 실패)")
        revised_opinion = Opinion(
            agent_id=self.agent_id,
            target=predicted_target,
            reason=revised_reason,
        )
        self.opinions.append(revised_opinion)
        return revised_opinion

    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        """Opinion 생성을 위한 LLM 메시지를 구성합니다"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_opinion method")

    def _build_messages_rebuttal(self, *args, **kwargs) -> Tuple[str, str]:
        """Rebuttal 생성을 위한 LLM 메시지를 구성합니다"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_rebuttal method")
    
    def _build_messages_revision(self, *args, **kwargs) -> Tuple[str, str]:
        """Revision 생성을 위한 LLM 메시지를 구성합니다"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_messages_revision method")

    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치를 로드합니다."""
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        try:
            # GPU 사용 가능 시 GPU로, 아니면 CPU로 로드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device)

            if getattr(self, "model", None) is None:
                if hasattr(self, "_build_model"):
                    self.model = self._build_model()
                elif hasattr(self, "forward"):
                    self.model = self
                else:
                    raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되어 있지 않습니다.")

            model = self.model
            
            if isinstance(checkpoint, torch.nn.Module):
                model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                model.load_state_dict(state_dict)
            else:
                print(f"알 수 없는 체크포인트 포맷: {type(checkpoint)}")
                return False

            # 모델을 GPU로 이동
            model = model.to(device)
            self.model = model
            model.eval()
            self.model_loaded = True
            print(f"[{self.agent_id}] 모델 로드 완료: {model_path} (device: {device})")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] 모델 로드 실패: {e}")
            return False

    def pretrain(self):
        """에이전트별 모델을 사전 학습합니다"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]

        X, y, cols = load_dataset(self.ticker, self.agent_id, save_dir=self.data_dir)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id} ({len(X)} samples)")

        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용")

        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_train = y * y_scale_factor
        X_train = X

        self.scaler.fit_scalers(X_train, y_train)
        self.scaler.save(self.ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_train, y_train))
        X_train, y_train = X_train.float(), y_train.float()

        if getattr(self, "model", None) is None:
            if hasattr(self, "_build_model"):
                self.model = self._build_model()
            else:
                raise RuntimeError(f"{self.agent_id}에 _build_model()이 정의되지 않음")

        model = self.model
        
        # GPU 사용 가능 시 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        loss_fn_name = agents_info.get(self.agent_id, {}).get("loss_fn", "HuberLoss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        elif loss_fn_name == "L1Loss":
            loss_fn = torch.nn.L1Loss()
        elif loss_fn_name == "MSELoss":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.HuberLoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in train_loader:
                y_pred = model(Xb)
                loss = loss_fn(y_pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.6f}")

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")
        torch.save({"model_state_dict": model.state_dict()}, model_path)
        self.model_loaded = True
        print(f"[{self.agent_id}] 모델 학습 및 저장 완료: {model_path} (device: {device})")

    def _ask_with_fallback(self, msg_sys: dict, msg_user: dict, schema_obj: dict) -> dict:
        """OpenAI API 호출"""
        if hasattr(self, 'test_mode') and self.test_mode:
            dummy_response = {}
            if schema_obj and isinstance(schema_obj, dict):
                props = schema_obj.get("properties", {})
                for key in props.keys():
                    if key == "reason":
                        dummy_response[key] = f"[백테스팅 모드] {self.agent_id} 예측 근거"
                    elif key == "stance":
                        dummy_response[key] = "SUPPORT"
                    elif key == "message":
                        dummy_response[key] = f"[백테스팅 모드] {self.agent_id} 메시지"
                    else:
                        prop_type = props[key].get("type", "string")
                        if prop_type == "string":
                            dummy_response[key] = ""
                        elif prop_type == "number":
                            dummy_response[key] = 0.0
                        else:
                            dummy_response[key] = None
            else:
                dummy_response = {"reason": f"[백테스팅 모드] {self.agent_id} 응답"}
            return dummy_response

        if not msg_sys or not msg_user:
            raise ValueError("Invalid messages: system or user message is None.")

        if schema_obj and isinstance(schema_obj, dict):
            schema_obj.setdefault("additionalProperties", False)
            if "type" not in schema_obj:
                schema_obj["type"] = "object"

        payload_base = {
            "input": [msg_sys, msg_user],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "Response",
                    "strict": True,
                    "schema": schema_obj,
                }
            },
            "temperature": self.temperature,
        }

        last_err = None
        for model in self.preferred_models:
            payload = dict(payload_base, model=model)
            try:
                r = requests.post(self.OPENAI_URL, headers=self.headers, json=payload, timeout=120)
                if r.ok:
                    data = r.json()
                    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
                        try:
                            return json.loads(data["output_text"])
                        except Exception:
                            return {"reason": data["output_text"]}
                    
                    out = data.get("output")
                    if isinstance(out, list) and out:
                        texts = []
                        for blk in out:
                            for c in blk.get("content", []):
                                if "text" in c:
                                    texts.append(c["text"])
                        joined = "\n".join(t for t in texts if t)
                        if joined.strip():
                            try:
                                return json.loads(joined)
                            except Exception:
                                return {"reason": joined}
                    return {}
                
                if r.status_code in (400, 404):
                    last_err = (r.status_code, r.text)
                    continue
                r.raise_for_status()
            except Exception as e:
                last_err = str(e)
                continue
        
        raise RuntimeError(f"모든 모델 실패. 마지막 오류: {last_err}")

    def _msg(self, role: str, content: str) -> dict:
        """OpenAI 메시지 포맷 생성"""
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"_msg() 인자 오류: role={role}, content={type(content)}")
        return {"role": role, "content": content}

    def _safe_float(self, v: Optional[float], default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


class DataScaler:
    """데이터 정규화를 담당하는 클래스"""

    def __init__(self, agent_id, scaler_dir: Optional[str] = None):
        self.agent_id = agent_id

        if scaler_dir is None:
            scaler_dir = dir_info.get(
                "scaler_dir",
                os.path.join(dir_info["model_dir"], "scalers")
            )
        self.save_dir = scaler_dir

        # 🔴 여기서는 "이름"만 보관
        self.x_scaler_name = agents_info[self.agent_id].get("x_scaler", "None")
        self.y_scaler_name = agents_info[self.agent_id].get("y_scaler", "None")

        # 🔴 실제 scaler 객체는 None으로 시작
        self.x_scaler = None
        self.y_scaler = None

    def fit_scalers(self, X_train, y_train):
        """스케일러를 학습합니다"""
        ScalerMap = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
            "None": None,
            None: None,
        }

        Sx = ScalerMap.get(self.x_scaler_name)
        Sy = ScalerMap.get(self.y_scaler_name)

        # ---------- X scaler ----------
        if Sx is not None:
            n_samples, seq_len, n_feats = X_train.shape
            X_2d = X_train.reshape(-1, n_feats)
            self.x_scaler = Sx().fit(X_2d)
        else:
            self.x_scaler = None

        # ---------- Y scaler ----------
        if Sy is not None:
            if Sy is MinMaxScaler:
                cfg = agents_info.get(self.agent_id, {})
                feature_range = cfg.get("minmax_scaler_range", (0, 1))
                self.y_scaler = Sy(feature_range=feature_range).fit(
                    y_train.reshape(-1, 1)
                )
            else:
                self.y_scaler = Sy().fit(y_train.reshape(-1, 1))
        else:
            self.y_scaler = None

    def transform(self, X, y=None):
        """데이터를 변환합니다"""
        # ---------- X ----------
        if self.x_scaler is not None:
            if X.ndim == 3:
                n_samples, seq_len, n_feats = X.shape
                X_2d = X.reshape(-1, n_feats)
                X_t = self.x_scaler.transform(X_2d).reshape(
                    n_samples, seq_len, n_feats
                )
            else:
                X_t = self.x_scaler.transform(X)
        else:
            X_t = X

        # ---------- Y ----------
        y_t = y
        if y is not None and self.y_scaler is not None:
            y_t = self.y_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_t, y_t

    def inverse_y(self, y_pred):
        """Y 값을 역변환합니다"""
        if self.y_scaler is None:
            return y_pred

        if isinstance(y_pred, (list, tuple)):
            y_pred = np.array(y_pred)

        return self.y_scaler.inverse_transform(
            y_pred.reshape(-1, 1)
        ).flatten()

    def save(self, ticker):
        """스케일러를 저장합니다"""
        os.makedirs(self.save_dir, exist_ok=True)

        if self.x_scaler is not None:
            joblib.dump(
                self.x_scaler,
                os.path.join(
                    self.save_dir,
                    f"{ticker}_{self.agent_id}_xscaler.pkl"
                )
            )

        if self.y_scaler is not None:
            joblib.dump(
                self.y_scaler,
                os.path.join(
                    self.save_dir,
                    f"{ticker}_{self.agent_id}_yscaler.pkl"
                )
            )

    def load(self, ticker):
        """스케일러를 로드합니다"""
        x_path = os.path.join(
            self.save_dir,
            f"{ticker}_{self.agent_id}_xscaler.pkl"
        )
        y_path = os.path.join(
            self.save_dir,
            f"{ticker}_{self.agent_id}_yscaler.pkl"
        )

        self.x_scaler = joblib.load(x_path) if os.path.exists(x_path) else None
        self.y_scaler = joblib.load(y_path) if os.path.exists(y_path) else None
