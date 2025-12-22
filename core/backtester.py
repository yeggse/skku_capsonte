import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import joblib
from datetime import datetime, timedelta

from agents.technical_agent import TechnicalAgent
from agents.macro_agent import MacroAgent
from agents.sentimental_agent import SentimentalAgent
from core.technical_classes.technical_data_set import load_dataset as load_dataset_tech
from config.agents_set import dir_info, agents_info


class Backtester:
    """통합 백테스팅 모듈"""
    def __init__(self, ticker, start_date=None, end_date=None, days=365, initial_capital=10000, commission=0.001):
        """백테스팅 모듈 초기화"""
        self.ticker = ticker.upper()
        self.initial_capital = initial_capital
        self.commission = commission
        
        if end_date is None:
            self.end_date = datetime.today()
        else:
            self.end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=days)
        else:
            self.start_date = pd.to_datetime(start_date)
            
        self.data_start_date = self.start_date - timedelta(days=365*2) 
        
        self.model = None
        self.full_data = None
        self.results = None
        self.available_features = None
        
        self.tech_agent = None
        self.macro_agent = None
        self.senti_agent = None

    def _init_agents(self):
        """에이전트를 초기화합니다"""
        print(f"[{self.ticker}] 에이전트 초기화 중...")
        
        # 1. TechnicalAgent
        self.tech_agent = TechnicalAgent(ticker=self.ticker)
        
        # 2. MacroAgent
        self.macro_agent = MacroAgent(ticker=self.ticker, base_date=datetime.today())
        
        # 3. SentimentalAgent
        self.senti_agent = SentimentalAgent(ticker=self.ticker)
        
        print("에이전트 초기화 완료. (pretrain은 각 날짜마다 수행됩니다)")

    def prepare_data(self):
        """데이터 수집 및 에이전트 예측값 생성 (Train/Test 데이터 확보)
        
        백테스팅 모드:
        1. 전체 데이터 1회 생성 (config 일자 + 테스트 기간)
        2. 각 날짜마다:
           - 슬라이싱 (해당 날짜 이전 데이터만)
           - pretrain (슬라이싱된 데이터로)
           - predict
        """
        if self.tech_agent is None:
            self._init_agents()

        print(f"[{datetime.now()}] 데이터 생성 및 예측 시작 ({self.data_start_date.date()} ~ {self.end_date.date()})")
        print("[INFO] 백테스팅 모드: 전체 데이터 1회 생성 후, 각 날짜마다 슬라이싱 → pretrain → predict")

        # ---------------------------------------------------------
        # 1. 가격 데이터 다운로드 (Target Dates 설정)
        # ---------------------------------------------------------
        df_price = yf.download(self.ticker, start=self.data_start_date, end=self.end_date, progress=False, auto_adjust=False)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0] for c in df_price.columns]
        df_price = df_price.reset_index()
        df_price['Date'] = pd.to_datetime(df_price['Date']).dt.normalize()
        df_price = df_price.sort_values('Date')
        
        target_dates = df_price['Date'].tolist()
        
        # ---------------------------------------------------------
        # 2. 전체 데이터 1회 생성 (config 일자 + 테스트 기간)
        # ---------------------------------------------------------
        print("[INFO] 전체 데이터 생성 중 (1회)...")
        
        # Technical Data 생성 (백테스팅 모드: data_start_date ~ end_date 기간으로 수집)
        tech_dataset_path = os.path.join(dir_info["data_dir"], f"{self.ticker}_TechnicalAgent_dataset.csv")
        # 백테스팅 모드에서는 data_start_date ~ end_date 기간의 데이터만 필요
        # 기존 데이터셋이 없거나, 필요한 기간을 포함하지 않으면 재생성
        need_rebuild = False
        if not os.path.exists(tech_dataset_path):
            need_rebuild = True
        
        if need_rebuild:
            # 백테스팅 모드: data_start_date ~ end_date 기간으로 데이터셋 생성
            from core.technical_classes.technical_data_set import build_dataset as build_dataset_tech
            cfg = agents_info.get("TechnicalAgent", {})
            # 필요한 기간 계산 (data_start_date ~ end_date)
            total_days = (self.end_date - self.data_start_date).days
            # period를 일수로 변환하여 전달 (최소 2년)
            period_days = max(total_days, 365*2)
            # yfinance의 period 형식으로 변환 (예: "2y")
            period_years = max(2, int(period_days / 365) + 1)  # 여유분 포함
            period_str = f"{period_years}y"
            print(f"[INFO] TechnicalAgent 데이터셋 생성: {self.data_start_date.date()} ~ {self.end_date.date()} (period={period_str})")
            build_dataset_tech(
                ticker=self.ticker,
                save_dir=dir_info["data_dir"],
                period=period_str,
                interval=cfg.get("interval", "1d"),
            )
        
        tech_X_all, tech_y_all, tech_cols, tech_dates = load_dataset_tech(
            self.ticker, agent_id="TechnicalAgent", save_dir=dir_info["data_dir"]
        )
        
        # tech_dates 평탄화
        tech_last_dates = []
        if tech_dates is not None and len(tech_dates) > 0:
            if isinstance(tech_dates[0], (list, tuple, np.ndarray)):
                tech_last_dates = [d[-1] for d in tech_dates]
            elif isinstance(tech_dates[0], str):
                tech_last_dates = tech_dates
        tech_last_dates_dt = pd.to_datetime(tech_last_dates).normalize()

        # Macro Data 생성 (백테스팅 모드: data_start_date ~ end_date 기간으로 수집)
        # MacroAgent의 searcher는 pretrain에서 호출되므로, 여기서는 직접 데이터 수집
        # 백테스팅 모드: data_start_date ~ end_date 기간으로 수집
        print(f"[INFO] MacroAgent 데이터 수집: {self.data_start_date.date()} ~ {self.end_date.date()}")
        # MacroAgent의 _fetch_macro_data와 _fetch_stock_data를 직접 호출하여 필요한 기간만 수집
        macro_df_all = self.macro_agent._fetch_macro_data(self.data_start_date, self.end_date)
        stock_df_all = self.macro_agent._fetch_stock_data(self.ticker, self.data_start_date, self.end_date)
        macro_df_all = self.macro_agent._add_derived_features(macro_df_all)
        _, macro_full_df_all = self.macro_agent._prepare_final_dataset(macro_df_all, stock_df_all, self.ticker)
        macro_full_df_all['Date'] = pd.to_datetime(macro_full_df_all['Date']).dt.normalize()

        # Sentiment Data 생성 (data_start_date ~ end_date 기간)
        days_needed = (self.end_date - self.data_start_date).days
        # 최소 2년치 확보
        days_needed = max(days_needed, 365*2)
        print(f"[INFO] SentimentalAgent 데이터 생성: {days_needed}일치 (약 {days_needed/365:.1f}년)")
        senti_sd_all = self.senti_agent.run_dataset(days=days_needed)
        senti_raw_all = senti_sd_all.raw_df.copy()
        senti_raw_all['date'] = pd.to_datetime(senti_raw_all['date']).dt.normalize()
        
        print("[INFO] 전체 데이터 생성 완료.")
        
        # ---------------------------------------------------------
        # 3. 각 날짜마다: 슬라이싱 → pretrain → predict
        # ---------------------------------------------------------
        results = []
        w_tech = self.tech_agent.window_size
        w_macro = self.macro_agent.window_size
        w_senti = self.senti_agent.window_size
        
        total_steps = len(target_dates)
        print(f"총 {total_steps}일 백테스팅 시작 (각 날짜마다 pretrain → predict)...")

        # 예측 성공/실패 통계
        stats = {
            'tech_success': 0, 'tech_fail': 0,
            'macro_success': 0, 'macro_fail': 0,
            'senti_success': 0, 'senti_fail': 0
        }

        for i, curr_date in enumerate(target_dates):
            if i % 50 == 0:
                print(f"  Processing {i}/{total_steps} ... (Date: {curr_date.date()})")
                
            price_row = df_price[df_price['Date'] == curr_date]
            if price_row.empty: 
                continue
            curr_close = float(price_row['Close'].iloc[0])
            idx_price = price_row.index[0]

            # Next Close (Target)
            if idx_price + 1 >= len(df_price):
                next_close_actual = np.nan
            else:
                next_close_actual = float(df_price.iloc[idx_price + 1]['Close'])

            # Look-ahead bias 방지: simulation_date 설정
            sim_date_str = curr_date.strftime("%Y-%m-%d")
            for agent in [self.tech_agent, self.macro_agent, self.senti_agent]:
                if hasattr(agent, 'set_test_mode'):
                    agent.set_test_mode(True)
                if hasattr(agent, 'set_simulation_date'):
                    agent.set_simulation_date(sim_date_str)

            # --- Technical Agent: 슬라이싱 → pretrain → predict ---
            pred_tech, conf_tech, unc_tech = np.nan, 0, 0
            try:
                # 1. 슬라이싱: simulation_date 이전 데이터만
                valid_tech_mask = tech_last_dates_dt <= curr_date
                tech_X_filtered = tech_X_all[valid_tech_mask.values] if hasattr(valid_tech_mask, 'values') else tech_X_all[valid_tech_mask]
                tech_y_filtered = tech_y_all[valid_tech_mask.values] if hasattr(valid_tech_mask, 'values') else tech_y_all[valid_tech_mask]
                tech_dates_filtered = tech_last_dates_dt[valid_tech_mask]
                
                if len(tech_X_filtered) < w_tech:
                    stats['tech_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] TechnicalAgent: {curr_date.date()}에 대한 충분한 데이터가 없습니다 (필요: {w_tech}, 사용가능: {len(tech_X_filtered)})")
                    raise ValueError("Insufficient data for pretrain")
                
                # 2. Pretrain (curr_date 이전 데이터로 전부 학습)
                # simulation_date가 설정되어 있으면 pretrain()이 자동으로 필터링함
                # 백테스팅 모드: 검증 데이터 분할 없이 전부 학습
                if len(tech_X_filtered) >= 30:  # 최소 학습 데이터 요구사항
                    try:
                        # TechnicalAgent의 pretrain은 simulation_date를 확인하여
                        # 해당 날짜 이전 데이터만 사용하여 학습함 (전부 학습, 검증 분할 없음)
                        self.tech_agent.pretrain()
                    except Exception as e:
                        if i < 5 or i % 50 == 0:
                            print(f"    [WARN] TechnicalAgent pretrain 실패 ({curr_date.date()}): {str(e)}")
                
                # 3. Predict (curr_date의 데이터로 다음날 종가 예측)
                # curr_date의 윈도우 데이터를 사용하여 다음날 종가 예측
                matches = np.where(tech_dates_filtered == curr_date)[0]
                if len(matches) > 0:
                    t_idx = matches[0]
                    X_batch = tech_X_filtered[t_idx] # (Win, F) - curr_date의 윈도우
                    X_in = np.expand_dims(X_batch, axis=0) # (1, Win, F)
                    # predict: curr_date의 데이터로 다음날 종가 예측
                    target_tech = self.tech_agent.predict(X_in, current_price=curr_close)
                    pred_tech = target_tech.next_close
                    conf_tech = target_tech.confidence
                    unc_tech = target_tech.uncertainty
                    ret_tech = getattr(target_tech, "predicted_return", (pred_tech - curr_close)/curr_close if not np.isnan(pred_tech) else np.nan)
                    stats['tech_success'] += 1
                else:
                    stats['tech_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] TechnicalAgent: {curr_date.date()}에 대한 데이터를 찾을 수 없습니다.")
            except Exception as e:
                stats['tech_fail'] += 1
                pred_tech, conf_tech, unc_tech, ret_tech = np.nan, 0, 0, np.nan
                if i < 5 or i % 100 == 0:
                    print(f"    [ERROR] TechnicalAgent 예측 실패 ({curr_date.date()}): {str(e)}")

            # --- Macro Agent: 슬라이싱 → pretrain → predict ---
            pred_macro, conf_macro, unc_macro, ret_macro = np.nan, 0, 0, np.nan
            try:
                # 1. 슬라이싱: simulation_date 이전 데이터만
                macro_full_df = macro_full_df_all[macro_full_df_all['Date'] <= curr_date].copy()
                
                if len(macro_full_df) < w_macro:
                    stats['macro_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] MacroAgent: {curr_date.date()}에 대한 충분한 데이터가 없습니다 (필요: {w_macro}, 사용가능: {len(macro_full_df)})")
                    raise ValueError("Insufficient data for pretrain")
                
                # 2. Pretrain (슬라이싱된 데이터로)
                # simulation_date가 설정되어 있으면 pretrain()이 자동으로 필터링함
                if len(macro_full_df) >= 30:  # 최소 학습 데이터 요구사항
                    try:
                        # MacroAgent의 pretrain은 simulation_date를 확인하여
                        # 해당 날짜 이전 데이터만 사용하여 학습함
                        self.macro_agent.pretrain()
                    except Exception as e:
                        if i < 5 or i % 50 == 0:
                            print(f"    [WARN] MacroAgent pretrain 실패 ({curr_date.date()}): {str(e)}")
                
                # 3. Predict
                m_match = macro_full_df[macro_full_df['Date'] == curr_date]
                if not m_match.empty:
                    m_idx = m_match.index[0]
                    if m_idx >= w_macro - 1 and len(macro_full_df) >= w_macro:
                        # 인덱스 재설정 후 슬라이싱
                        macro_full_df = macro_full_df.reset_index(drop=True)
                        m_idx_new = len(macro_full_df) - 1
                        df_slice = macro_full_df.iloc[m_idx_new - w_macro + 1 : m_idx_new + 1]
                        
                        # scaler_X가 없으면 스킵
                        if not hasattr(self.macro_agent, 'scaler_X') or self.macro_agent.scaler_X is None:
                            stats['macro_fail'] += 1
                            if i < 5 or i % 100 == 0:
                                print(f"    [WARN] MacroAgent: scaler_X가 초기화되지 않았습니다.")
                            raise ValueError("MacroAgent scaler_X not initialized")
                        
                        feat_cols = list(self.macro_agent.scaler_X.feature_names_in_)
                        X_slice = pd.DataFrame(index=df_slice.index)
                        for c in feat_cols:
                            X_slice[c] = df_slice[c] if c in df_slice.columns else 0.0
                        
                        X_sc = self.macro_agent.scaler_X.transform(X_slice)
                        X_in = np.expand_dims(X_sc, axis=0)
                        X_tensor = torch.FloatTensor(X_in).to(self.macro_agent.device)
                        target_macro = self.macro_agent.predict(X_tensor, current_price=curr_close)
                        pred_macro = target_macro.next_close
                        conf_macro = target_macro.confidence
                        unc_macro = target_macro.uncertainty
                        ret_macro = getattr(target_macro, "predicted_return", (pred_macro - curr_close)/curr_close if not np.isnan(pred_macro) else np.nan)
                        stats['macro_success'] += 1
                    else:
                        stats['macro_fail'] += 1
                        if i < 5 or i % 100 == 0:
                            print(f"    [WARN] MacroAgent: {curr_date.date()}에 대한 충분한 윈도우 데이터가 없습니다 (필요: {w_macro}, 사용가능: {len(macro_full_df)})")
                else:
                    stats['macro_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] MacroAgent: {curr_date.date()}에 대한 데이터를 찾을 수 없습니다.")
            except Exception as e:
                stats['macro_fail'] += 1
                pred_macro, conf_macro, unc_macro, ret_macro = np.nan, 0, 0, np.nan
                if i < 5 or i % 100 == 0:
                    print(f"    [ERROR] MacroAgent 예측 실패 ({curr_date.date()}): {str(e)}")

            # --- Sentimental Agent: 슬라이싱 → pretrain → predict ---
            pred_senti, conf_senti, unc_senti, ret_senti = np.nan, 0, 0, np.nan
            try:
                # 1. 슬라이싱: simulation_date 이전 데이터만
                senti_raw = senti_raw_all[senti_raw_all['date'] <= curr_date].copy()
                senti_feat_values = senti_raw[senti_sd_all.feature_cols].astype(float).values
                
                if len(senti_raw) < w_senti:
                    stats['senti_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] SentimentalAgent: {curr_date.date()}에 대한 충분한 데이터가 없습니다 (필요: {w_senti}, 사용가능: {len(senti_raw)})")
                    raise ValueError("Insufficient data for pretrain")
                
                # 2. Pretrain (슬라이싱된 데이터로)
                # simulation_date가 설정되어 있으면 pretrain()이 자동으로 필터링함
                if len(senti_raw) >= 30:  # 최소 학습 데이터 요구사항
                    try:
                        # SentimentalAgent의 pretrain은 simulation_date를 확인하여
                        # 해당 날짜 이전 데이터만 사용하여 학습함
                        self.senti_agent.pretrain()
                    except Exception as e:
                        if i < 5 or i % 50 == 0:
                            print(f"    [WARN] SentimentalAgent pretrain 실패 ({curr_date.date()}): {str(e)}")
                
                # 3. Predict
                s_match = senti_raw[senti_raw['date'] == curr_date]
                if not s_match.empty:
                    s_idx = s_match.index[0]
                    if s_idx >= w_senti - 1 and len(senti_raw) >= w_senti:
                        # 인덱스 재설정 후 슬라이싱
                        senti_raw = senti_raw.reset_index(drop=True)
                        s_idx_new = len(senti_raw) - 1
                        X_batch = senti_feat_values[s_idx_new - w_senti + 1 : s_idx_new + 1]
                        X_in = np.expand_dims(X_batch, axis=0)
                        target_senti = self.senti_agent.predict(X_in, current_price=curr_close)
                        pred_senti = target_senti.next_close
                        conf_senti = target_senti.confidence
                        unc_senti = target_senti.uncertainty
                        ret_senti = getattr(target_senti, "predicted_return", (pred_senti - curr_close)/curr_close if not np.isnan(pred_senti) else np.nan)
                        stats['senti_success'] += 1
                    else:
                        stats['senti_fail'] += 1
                        if i < 5 or i % 100 == 0:
                            print(f"    [WARN] SentimentalAgent: {curr_date.date()}에 대한 충분한 윈도우 데이터가 없습니다 (필요: {w_senti}, 사용가능: {len(senti_raw)})")
                else:
                    stats['senti_fail'] += 1
                    if i < 5 or i % 100 == 0:
                        print(f"    [WARN] SentimentalAgent: {curr_date.date()}에 대한 데이터를 찾을 수 없습니다.")
            except Exception as e:
                stats['senti_fail'] += 1
                pred_senti, conf_senti, unc_senti, ret_senti = np.nan, 0, 0, np.nan
                if i < 5 or i % 100 == 0:
                    print(f"    [ERROR] SentimentalAgent 예측 실패 ({curr_date.date()}): {str(e)}")

            results.append({
                "Date": curr_date,
                "Last_Close": curr_close,
                "Next_Close": next_close_actual,
                "Tech_Pred": pred_tech, "Tech_Conf": conf_tech, "Tech_Unc": unc_tech, "Tech_Ret": ret_tech,
                "Macro_Pred": pred_macro, "Macro_Conf": conf_macro, "Macro_Unc": unc_macro, "Macro_Ret": ret_macro,
                "Senti_Pred": pred_senti, "Senti_Conf": conf_senti, "Senti_Unc": unc_senti, "Senti_Ret": ret_senti
            })
        
        # 결과 통계 출력
        print(f"\n[예측 통계]")
        print(f"  TechnicalAgent: 성공 {stats['tech_success']}회, 실패 {stats['tech_fail']}회")
        print(f"  MacroAgent: 성공 {stats['macro_success']}회, 실패 {stats['macro_fail']}회")
        print(f"  SentimentalAgent: 성공 {stats['senti_success']}회, 실패 {stats['senti_fail']}회")
        
        # 최소 하나의 예측값이 있는 행만 유지
        self.full_data = pd.DataFrame(results)
        # 모든 예측값이 NaN인 행 제거
        pred_cols = ['Tech_Pred', 'Macro_Pred', 'Senti_Pred']
        valid_mask = self.full_data[pred_cols].notna().any(axis=1)
        self.full_data = self.full_data[valid_mask]
        
        print(f"데이터 생성 완료: {len(self.full_data)}행 (전체 {len(results)}행 중 유효한 행)")
        
        if len(self.full_data) == 0:
            print("[ERROR] 유효한 예측 데이터가 없습니다. 에이전트 설정 및 데이터 수집을 확인하세요.")

    def train_model(self):
        """확보된 데이터의 앞부분(Train)으로 LightGBM 앙상블 모델 학습
        Look-ahead bias 방지를 위해 날짜 기준으로 엄격하게 train/test를 분리합니다.
        """
        if self.full_data is None or len(self.full_data) < 10:
            print("[WARN] 데이터가 부족하여 학습을 건너뜁니다. (최소 10행 필요)")
            return

        df = self.full_data.copy()
        
        # Feature Engineering (수익률 변환)
        # 데이터셋에 이미 *_Ret 컬럼이 있으면 사용, 없으면 계산
        if 'Tech_Ret' not in df.columns:
            df['Tech_Ret'] = (df['Tech_Pred'] - df['Last_Close']) / df['Last_Close']
        else:
            df['Tech_Ret'] = df['Tech_Ret'].fillna((df['Tech_Pred'] - df['Last_Close']) / df['Last_Close'])

        if 'Macro_Ret' not in df.columns:
            df['Macro_Ret'] = (df['Macro_Pred'] - df['Last_Close']) / df['Last_Close']
        else:
            df['Macro_Ret'] = df['Macro_Ret'].fillna((df['Macro_Pred'] - df['Last_Close']) / df['Last_Close'])

        if 'Senti_Ret' not in df.columns:
            df['Senti_Ret'] = (df['Senti_Pred'] - df['Last_Close']) / df['Last_Close']
        else:
            df['Senti_Ret'] = df['Senti_Ret'].fillna((df['Senti_Pred'] - df['Last_Close']) / df['Last_Close'])

        df['Target_Ret'] = (df['Next_Close'] - df['Last_Close']) / df['Last_Close']
        
        feature_cols = [
            'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
            'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
            'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
        ]
        
        # Train/Test Split (날짜 기준 - self.start_date가 이미 테스트 시작 시점으로 정해져 있음)
        # self.start_date 이전 데이터는 학습용, 이후는 테스트용
        df = df.sort_values('Date')  # 날짜 순서 보장
        train_mask = df['Date'] < self.start_date
        test_mask = df['Date'] >= self.start_date
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'Target_Ret']
        
        # NaN 값이 있는 행 제거 (모든 feature가 유효한 행만 사용)
        train_valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[train_valid_mask]
        y_train = y_train[train_valid_mask]
        
        # 최소 데이터 요구사항 확인
        MIN_TRAIN_SAMPLES = 30
        if len(X_train) < MIN_TRAIN_SAMPLES:
            print(f"[WARN] 학습 데이터가 부족합니다 (현재: {len(X_train)}행, 최소 필요: {MIN_TRAIN_SAMPLES}행)")
            print(f"  - 전체 데이터: {len(df)}행")
            print(f"  - 학습 기간: {df.loc[train_mask, 'Date'].min().date()} ~ {df.loc[train_mask, 'Date'].max().date() if train_mask.any() else 'N/A'}")
            print(f"  - 테스트 기간: {self.start_date.date()} ~ {df.loc[test_mask, 'Date'].max().date() if test_mask.any() else 'N/A'}")
            print(f"[ERROR] 학습 데이터가 부족하여 학습을 건너뜁니다.")
            return
        
        print(f"Meta Model 학습 시작")
        print(f"  - Train size: {len(X_train)}행")
        print(f"  - Train 기간: {df.loc[train_mask, 'Date'].min().date()} ~ {df.loc[train_mask, 'Date'].max().date()}")
        print(f"  - Test 기간: {df.loc[test_mask, 'Date'].min().date() if test_mask.any() else 'N/A'} ~ {df.loc[test_mask, 'Date'].max().date() if test_mask.any() else 'N/A'}")
        
        # Custom Objective Function Import
        from scripts.train_meta_model import directional_mse_objective
        from config.agents_set import common_params

        self.model = lgb.LGBMRegressor(
            n_estimators=common_params.get("ensemble_n_estimators", 100),
            learning_rate=common_params.get("ensemble_learning_rate", 0.05),
            max_depth=common_params.get("ensemble_max_depth", 3),
            random_state=common_params.get("ensemble_random_state", 42),
            n_jobs=common_params.get("ensemble_n_jobs", -1),
            objective=directional_mse_objective
        )
        
        self.model.fit(X_train, y_train)
        print("Meta Model 학습 완료.")

    def run_simulation(self, buy_threshold=0.005, sell_threshold=0.005):
        """
        Backtest Simulation 실행
        :param buy_threshold: 예측 상승률이 이 값보다 크면 매수 (0.005 = 0.5%)
        :param sell_threshold: 예측 하락률이 이 값보다 크면 매도
        """
        if self.full_data is None or self.full_data.empty:
            print("[INFO] 데이터 준비 중...")
            self.prepare_data()
            
        if self.full_data is None or self.full_data.empty:
            print("[ERROR] 데이터 준비에 실패했습니다. 시뮬레이션을 중단합니다.")
            return
            
        if self.model is None:
            print("[INFO] 모델 학습 중...")
            self.train_model()
            
        if self.model is None:
            print("[ERROR] 모델 학습에 실패했습니다. 시뮬레이션을 중단합니다.")
            return
            
        # 테스트 기간 데이터 필터링
        test_df = self.full_data[self.full_data['Date'] >= self.start_date].copy().reset_index(drop=True)
        
        if test_df.empty:
            print(f"[ERROR] 테스트 기간({self.start_date.date()} ~ {self.end_date.date()})에 해당하는 데이터가 없습니다.")
            print(f"  - 전체 데이터 기간: {self.full_data['Date'].min().date()} ~ {self.full_data['Date'].max().date()}")
            return

        # 피처 생성
        test_df['Tech_Ret'] = (test_df['Tech_Pred'] - test_df['Last_Close']) / test_df['Last_Close']
        test_df['Macro_Ret'] = (test_df['Macro_Pred'] - test_df['Last_Close']) / test_df['Last_Close']
        test_df['Senti_Ret'] = (test_df['Senti_Pred'] - test_df['Last_Close']) / test_df['Last_Close']
        
        feature_cols = [
            'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
            'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
            'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
        ]
        
        # NaN 값이 있는 행 제거 (모든 feature가 유효한 행만 사용)
        valid_mask = test_df[feature_cols].notna().all(axis=1)
        test_df = test_df[valid_mask].copy().reset_index(drop=True)
        
        if test_df.empty:
            print("[ERROR] 유효한 피처 데이터가 없습니다. 시뮬레이션을 중단합니다.")
            return
        
        # 앙상블 예측
        try:
            pred_rets = self.model.predict(test_df[feature_cols])
            test_df['Ensemble_Ret'] = pred_rets
            test_df['Ensemble_Pred'] = test_df['Last_Close'] * (1 + pred_rets)
        except Exception as e:
            print(f"[ERROR] 앙상블 예측 실패: {str(e)}")
            return
        
        # --- Simulation Loop ---
        cash = self.initial_capital
        shares = 0
        portfolio_values = []
        trades = []
        
        print(f"시뮬레이션 시작: {len(test_df)}일")
        
        if len(test_df) == 0:
            print("[ERROR] 시뮬레이션할 데이터가 없습니다.")
            self.results = pd.DataFrame()
            self.trades = pd.DataFrame()
            return
        
        for i, row in test_df.iterrows():
            curr_date = row['Date']
            curr_price = row['Last_Close']
            pred_ret = row['Ensemble_Ret']
            
            action = "HOLD"
            
            # 매수 조건
            if pred_ret > buy_threshold and cash > curr_price:
                # 전량 매수 (간단한 로직)
                buyable_shares = int(cash / (curr_price * (1 + self.commission)))
                if buyable_shares > 0:
                    cost = buyable_shares * curr_price
                    fee = cost * self.commission
                    cash -= (cost + fee)
                    shares += buyable_shares
                    action = "BUY"
                    trades.append({
                        "Date": curr_date, "Type": "BUY", "Price": curr_price, 
                        "Shares": buyable_shares, "Value": cost, "Fee": fee
                    })
            
            # 매도 조건
            elif (pred_ret < -sell_threshold) and shares > 0:
                # 전량 매도
                revenue = shares * curr_price
                fee = revenue * self.commission
                cash += (revenue - fee)
                shares = 0
                action = "SELL"
                trades.append({
                    "Date": curr_date, "Type": "SELL", "Price": curr_price, 
                    "Shares": revenue/curr_price, "Value": revenue, "Fee": fee
                })
                
            # 포트폴리오 가치 기록
            total_value = cash + (shares * curr_price)
            portfolio_values.append({
                "Date": curr_date,
                "Total_Value": total_value,
                "Cash": cash,
                "Stock_Value": shares * curr_price,
                "Shares": shares,
                "Price": curr_price,
                "Action": action
            })
            
        self.results = pd.DataFrame(portfolio_values)
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["Date", "Type", "Price", "Shares", "Value", "Fee"])
        print(f"시뮬레이션 종료. (거래 횟수: {len(self.trades)}회)")

    def calculate_metrics(self):
        """수익률, MDD, 승률 등 지표를 계산합니다"""
        if self.results is None or self.results.empty:
            print("[WARN] 결과 데이터가 없어 지표를 계산할 수 없습니다.")
            return {}
            
        df = self.results.copy()
        
        initial_val = self.initial_capital
        final_val = df.iloc[-1]['Total_Value']
        
        # 1. 누적 수익률
        total_return = (final_val - initial_val) / initial_val * 100
        
        # 2. MDD (Maximum Drawdown)
        df['Peak'] = df['Total_Value'].cummax()
        df['Drawdown'] = (df['Total_Value'] - df['Peak']) / df['Peak'] * 100
        mdd = df['Drawdown'].min()
        
        # 3. 승률 및 매매 통계
        trade_count = 0
        if hasattr(self, 'trades') and self.trades is not None and not self.trades.empty:
            # 매도 기록만 추출 (매도가 실제 거래 완료를 의미)
            sells = self.trades[self.trades['Type'] == 'SELL']
            trade_count = len(sells)
        else:
            # trades가 없으면 매수/매도 액션 수로 계산
            if 'Action' in df.columns:
                trade_count = len(df[df['Action'].isin(['BUY', 'SELL'])])
        
        # 4. 일별 수익률 통계
        df['Daily_Return'] = df['Total_Value'].pct_change() * 100
        avg_daily_return = df['Daily_Return'].mean()
        volatility = df['Daily_Return'].std()

        return {
            "Initial_Capital": initial_val,
            "Final_Capital": final_val,
            "Total_Return": total_return,
            "MDD": mdd,
            "Trade_Count": trade_count,
            "Avg_Daily_Return": avg_daily_return,
            "Volatility": volatility
        }

    def save_results(self, output_dir="data/backtest"):
        """결과 CSV 및 그래프를 저장합니다"""
        if self.results is None or self.results.empty:
            print("[WARN] 저장할 결과 데이터가 없습니다. 시뮬레이션이 실행되지 않았거나 실패했습니다.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # CSV 저장
            res_path = os.path.join(output_dir, f"{self.ticker}_backtest_log.csv")
            self.results.to_csv(res_path, index=False)
            print(f"[OK] 결과 CSV 저장 완료: {res_path}")
        except Exception as e:
            print(f"[ERROR] CSV 저장 실패: {str(e)}")
            return
        
        try:
            # 그래프 그리기
            plt.figure(figsize=(12, 6))
            plt.plot(self.results['Date'], self.results['Total_Value'], label='Portfolio Value')
            
            # Buy & Hold 비교
            first_price = self.results.iloc[0]['Price']
            buy_hold_shares = self.initial_capital / first_price
            self.results['Buy_Hold'] = self.results['Price'] * buy_hold_shares
            plt.plot(self.results['Date'], self.results['Buy_Hold'], label='Buy & Hold', linestyle='--', alpha=0.7)
            
            plt.title(f"Backtest Result: {self.ticker}")
            plt.xlabel("Date")
            plt.ylabel("Value ($)")
            plt.legend()
            plt.grid(True)
            
            img_path = os.path.join(output_dir, f"{self.ticker}_backtest_chart.png")
            plt.savefig(img_path)
            plt.close()
            print(f"[OK] 차트 저장 완료: {img_path}")
        except Exception as e:
            print(f"[ERROR] 차트 저장 실패: {str(e)}")

