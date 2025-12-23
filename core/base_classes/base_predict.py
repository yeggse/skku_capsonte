from typing import Optional

import numpy as np
import torch

from agents.base_agent import Target, StockData
from config.agents_set import common_params


class BaseAgentPredictMixin:

    def predict(self, input_data=None, *, current_price: Optional[float] = None, return_raw: bool = False) -> Target:

        # --------------------------------------------------
        # 1. 입력 데이터 준비
        # --------------------------------------------------
        if input_data is None:
            X = self.search(self.ticker)
        elif isinstance(input_data, torch.Tensor):
            X = input_data
        elif isinstance(input_data, np.ndarray):
            # ★ 앙상블/백테스트용 입력
            X = torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, StockData):
            X = self.search(self.ticker)
        else:
            raise TypeError(f"predict(): 지원하지 않는 input 타입 {type(input_data)}")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # --------------------------------------------------
        # 2. 모델 결정
        # --------------------------------------------------
        model = getattr(self, "model", None)
        if model is None:
            if isinstance(self, torch.nn.Module):
                model = self
            else:
                raise RuntimeError(f"[{self.agent_id}] model이 설정되지 않았습니다.")

        # --------------------------------------------------
        # 3. 디바이스 설정
        # --------------------------------------------------
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        X = X.to(device).float()

        # --------------------------------------------------
        # 4. Forward (scaled return)
        # --------------------------------------------------
        model.eval()
        with torch.no_grad():
            out = model(X)
            if isinstance(out, (tuple, list)):
                out = out[0]

            y_pred_scaled = out.detach().cpu().numpy().reshape(-1)

        # --------------------------------------------------
        # 5. 역스케일링 (모델 출력)
        # --------------------------------------------------
        if hasattr(self, "scaler") and getattr(self.scaler, "y_scaler", None) is not None:
            try:
                y_pred = self.scaler.inverse_y(y_pred_scaled)
            except Exception:
                y_pred = y_pred_scaled
        else:
            y_pred = y_pred_scaled

        y_pred = float(y_pred[0])

        # --------------------------------------------------
        # 6. return / price 분기 (유일한 분기 지점)
        # --------------------------------------------------
        mode = getattr(self, "predict_mode", "return")

        current_price = self._extract_current_price(
            input_data=input_data,
            input_array=X.detach().cpu().numpy(),
            current_price=current_price,
        )

        if mode == "price": # SentimentalAgent는 ‘수익률 예측 모델’이 아니라 ‘가격(또는 score) 직접 예측 모델’ 구조
            # SentimentalAgent
            next_close = y_pred
            predicted_return = (next_close / current_price) - 1.0
        else:
            # Technical / Macro
            y_scale_factor = common_params.get("y_scale_factor", 100.0)
            predicted_return = y_pred / y_scale_factor
            next_close = current_price * (1.0 + predicted_return)




        # --------------------------------------------------
        # 7. confidence (None-safe)
        # --------------------------------------------------
        try:
            confidence = self._calculate_confidence_from_direction_accuracy()
        except Exception:
            confidence = None

        if confidence is None:
            confidence = 0.5

        # --------------------------------------------------
        # 8. uncertainty (None-safe)
        # --------------------------------------------------
        sigma_scale = common_params.get("uncertainty_scale", None)
        if sigma_scale is not None:
            uncertainty = abs(predicted_return) * sigma_scale
        else:
            uncertainty = 0.0

        # --------------------------------------------------
        # 9. 결과 반환
        # --------------------------------------------------
        if return_raw:
            return {
                "predicted_return": predicted_return,
                "next_close": float(next_close),
                "confidence": confidence,
                "uncertainty": uncertainty,
            }

        return Target(
            next_close=float(next_close),
            uncertainty=float(uncertainty),
            confidence=float(confidence),
            predicted_return=float(predicted_return),
        )
