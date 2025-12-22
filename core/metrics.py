
import numpy as np
import pandas as pd
from typing import Dict, Union, List

def calculate_metrics(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    기본 예측 성능 지표 계산 (MAE, RMSE, MAPE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 길이가 다르면 최소 길이로 맞춤
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    #MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE (y_true가 0인 경우 제외하거나 작은 값 더함)
    # 여기서는 0인 경우 해당 샘플 제외 처리
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = 0.0
        
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }

def calculate_direction_accuracy(y_true: Union[np.ndarray, List[float]], y_pred: Union[np.ndarray, List[float]], prev_close: Union[np.ndarray, List[float]]) -> float:
    """
    방향 정확도 계산
    - 실제 등락 부호와 예측 등락 부호가 일치하는 비율
    - y_true, y_pred가 '가격'인 경우 prev_close가 필요함
    - y_true, y_pred가 '수익률'인 경우 prev_close는 무시 가능 (부호만 보면 됨)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    prev_close = np.array(prev_close)
    
    min_len = min(len(y_true), len(y_pred), len(prev_close))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    prev_close = prev_close[:min_len]
    
    # 실제 변동
    actual_diff = y_true - prev_close
    # 예측 변동
    pred_diff = y_pred - prev_close
    
    # 부호 일치 여부 (0은 변동 없음으로 처리, 여기서는 부호가 같거나 하나가 0이면 제외? 
    # 단순하게 sign * sign > 0 으로 판별)
    
    # sign이 0인 경우는 방향성 판단이 모호하므로 제외하거나 맞았다고 칠 수 있음.
    # 여기서는 엄격하게 sign이 같아야 한다고 가정 (0 제외)
    
    actual_sign = np.sign(actual_diff)
    pred_sign = np.sign(pred_diff)
    
    # 둘 다 0이면 맞음, 그 외엔 부호 같아야 함
    match = (actual_sign == pred_sign)
    
    return float(np.mean(match) * 100.0)

def calculate_profitability(
    dates: List[str],
    actual_prices: List[float],
    predicted_prices: List[float],
    initial_capital: float = 10000.0
) -> Dict[str, float]:
    """
    간단한 전략 수익률 시뮬레이션
    - 전략: 예측가가 현재가(전일 종가)보다 높으면 매수/보유, 낮으면 매도/무포지션 (Long Only)
    - 벤치마크: Buy & Hold
    """
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    
    if len(actual_prices) == 0 or len(actual_prices) < 2:
        return {"Strategy_Return": 0.0, "BuyHold_Return": 0.0}
        
    capital = initial_capital
    shares = 0.0
    
    # Buy & Hold (첫날 매수, 마지막날 평가)
    bh_shares = initial_capital / actual_prices[0]
    bh_final_value = bh_shares * actual_prices[-1]
    bh_return = (bh_final_value / initial_capital - 1) * 100.0
    
    # Strategy
    # Day 0: 관망 (데이터 필요)
    # Day i: Day i-1의 예측(Day i 종가 예측)을 보고 시가(또는 전일 종가)에 매매 결정한다고 가정
    # 여기서는 간단히: 
    # i번째 날의 예측값(predicted_prices[i])은 i번째 날의 종가를 예측한 것.
    # 매매 판단 시점: i-1번째 날 장 마감 후 또는 i번째 날 장 시작 시.
    # 거래 가격: i-1번째 날 종가(=i번째 날 시가 근사)로 가정하거나 i번째 날 실제 종가로 수익률 계산.
    # Rolling Backtest 구조상 predicted_prices[i]는 T-1 시점에 T 시점을 예측한 값임.
    # 따라서 T-1 시점의 종가(actual_prices[i-1])와 비교하여 상승 예측이면 T 시점 보유.
    
    capital_curve = [initial_capital]
    
    # 첫 날은 전일 데이터가 없으므로 포지션 0에서 시작한다고 가정
    # 실제로는 loop index 1부터 시작
    
    current_capital = initial_capital
    
    for i in range(1, len(actual_prices)):
        prev_close = actual_prices[i-1]
        curr_close = actual_prices[i]
        pred_close = predicted_prices[i] # T-1 시점에 예측한 T의 종가
        
        # 전략: 상승 예측 시 보유 (Long)
        if pred_close > prev_close:
            # 수익률 적용 (오늘 종가 / 어제 종가)
            daily_ret = (curr_close / prev_close)
            current_capital *= daily_ret
        else:
            # 하락/보합 예측 시 현금 보유 (수익률 1.0)
            pass
            
        capital_curve.append(current_capital)
        
    strategy_return = (current_capital / initial_capital - 1) * 100.0
    
    return {
        "Strategy_Return": strategy_return,
        "BuyHold_Return": bh_return,
        "Final_Capital": current_capital
    }



