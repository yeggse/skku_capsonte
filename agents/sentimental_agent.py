# agents/sentimental_agent.py

from __future__ import annotations

import os
import json
from typing import Optional, Tuple, Dict, Any, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset

from agents.base_agent import BaseAgent, StockData, Target, Opinion, Rebuttal
from core.sentimental_classes.news import merge_price_with_news_features
from core.sentimental_classes.lstm_model import SentimentalLSTM
from core.data_set import load_dataset, build_dataset
from config.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS
from config.agents_set import agents_info

from config.agents_set import common_params
from core.sentimental_classes.news import update_news_db

CFG_S = agents_info["SentimentalAgent"]
WINDOW_SIZE = CFG_S["window_size"]
HIDDEN_DIM = CFG_S.get("d_model", 64)
NUM_LAYERS = CFG_S["num_layers"]
DROPOUT = CFG_S["dropout"]


class SentimentalAgent(BaseAgent):
    """뉴스 기사와 감성 점수를 기반으로 주가를 예측하는 에이전트"""

    def __init__(self, ticker, agent_id="SentimentalAgent", news_dir=None, **kwargs):
        """SentimentalAgent 초기화"""
        super().__init__(ticker=ticker, agent_id=agent_id, **kwargs)
        
        if news_dir is None:
            base_dir = os.path.dirname(self.data_dir)
            news_dir = os.path.join(base_dir, "raw", "news")
        self.news_dir = news_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = agents_info[self.agent_id]
        self.window_size = cfg["window_size"]
        self.hidden_dim = HIDDEN_DIM
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT
        self.feature_cols = CFG_S.get("data_cols", [])

        self.model = None
        self.model_loaded = False

        if not getattr(self, "ticker", None):
            self.ticker = ticker
        if not self.ticker:
            raise ValueError("SentimentalAgent: ticker is None/empty")
        self.ticker = str(self.ticker).upper()
        setattr(self, "ticker", self.ticker)


    def pretrain(self):
        """사전학습을 수행합니다"""
        epochs = agents_info[self.agent_id]["epochs"]
        lr = agents_info[self.agent_id]["learning_rate"]
        batch_size = agents_info[self.agent_id]["batch_size"]
        
        if not self.ticker:
            raise ValueError("SentimentalAgent.pretrain: ticker가 설정되지 않았습니다.")
        
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
        
        feature_cols = CFG_S.get("data_cols", [])
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
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중, 마지막 타겟 제거 (데이터 누수 방지)")
            else:
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중 (타겟 없음)")
        
        # 3) 시퀀스 생성
        window_size = self.window_size
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        def _create_sequences(X, y, win: int):
            Xs, ys = [], []
            for i in range(len(X) - win):
                Xs.append(X[i : i + win])
                ys.append(y[i + win])
            return np.array(Xs), np.array(ys)
        
        X_seq, y_seq = _create_sequences(X_all, y_all, window_size)
        print(f"[INFO] 시퀀스 생성 완료: {X_seq.shape}, {y_seq.shape}")
        
        # 4) 타깃 스케일 조정 (수익률 * 100)
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_seq = y_seq * y_scale_factor
        
        # 5) 스케일링
        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(ticker)
        
        X_train, y_train = map(torch.tensor, self.scaler.transform(X_seq, y_seq))
        X_train, y_train = X_train.float(), y_train.float()
        
        # 6) 모델 생성 및 학습
        if getattr(self, "model", None) is None:
            input_dim = X_seq.shape[-1]
            self.model = SentimentalLSTM(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        
        model = self.model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        cfg = agents_info.get(self.agent_id, {})
        loss_fn_name = cfg.get("loss_fn", "HuberLoss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = torch.nn.HuberLoss(delta=huber_delta)
        elif loss_fn_name == "L1Loss":
            loss_fn = torch.nn.L1Loss()
        elif loss_fn_name == "MSELoss":
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = torch.nn.HuberLoss(delta=common_params.get("huber_loss_delta", 1.0))
        
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
        best_val_loss_orig = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            total_loss = 0.0
            total_loss_original = 0.0
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
        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {model_path}{final_loss_str}")
        
        if common_params.get("pretrain_save_dataset", True):
            dataset_path = os.path.join(self.data_dir, f"{ticker}_{self.agent_id}_dataset.csv")
            flattened_data = []
            dates_list = df_raw["Date"].values[:-1]
            
            X_scaled_np = X_train.cpu().numpy()
            y_scaled_np = y_train.cpu().numpy()
            
            for sample_idx in range(len(X_seq)):
                for time_idx in range(window_size):
                    date_idx = sample_idx + time_idx
                    row = {
                        'sample_id': sample_idx,
                        'time_step': time_idx,
                        'date': str(dates_list[date_idx]) if date_idx < len(dates_list) else None,
                        'target': float(y_scaled_np[sample_idx]) if time_idx == window_size - 1 else np.nan,
                    }
                    for feat_idx, feat_name in enumerate(feature_cols):
                        row[feat_name] = float(X_scaled_np[sample_idx, time_idx, feat_idx])
                    flattened_data.append(row)
            
            dataset_df = pd.DataFrame(flattened_data)
            os.makedirs(self.data_dir, exist_ok=True)
            dataset_df.to_csv(dataset_path, index=False)
            print(f"✅ 전처리된 데이터 저장 완료: {dataset_path}")

    def _build_model(self) -> nn.Module:
        """LSTM 모델을 생성합니다"""
        try:
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
        except Exception:
            build_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )
            X, y, cols = load_dataset(
                ticker=self.ticker,
                agent_id=self.agent_id,
            )

        self.feature_cols = list(cols)
        data_cols = CFG_S.get("data_cols", [])
        input_dim = CFG_S.get("input_dim", len(data_cols) if data_cols else 8)

        model = SentimentalLSTM(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return model

    # -------------------------------------------------------
    # RUN_DATASET
    # -------------------------------------------------------
    def run_dataset(self, days: int = None) -> StockData:
        """
        최근 days일치 가격 + 뉴스 피처를 기반으로 StockData를 생성합니다.
        """
        if days is None:
            # 백테스팅 여부에 따라 기간 설정
            if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
                period_str = common_params.get("period_test", "2y")
            else:
                period_str = common_params.get("period", "2y")
            
            # 문자열 기간 -> 일수 변환
            if period_str.endswith("y"):
                years = int(period_str[:-1])
                days = years * 365
            elif period_str.endswith("m"):
                months = int(period_str[:-1])
                days = months * 30
            elif period_str.endswith("d"):
                days = int(period_str[:-1])
            else:
                days = 2 * 365
        
        # 0) 날짜 범위
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # 1) 가격 데이터 수집
        df_price = yf.download(self.ticker, start=start, end=end, auto_adjust=False)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # 2) 뉴스 + 가격 병합
        df_merged = merge_price_with_news_features( 
            df_price=df_price,
            ticker=self.ticker,
            asof_kst=end.date(),
            base_dir=self.news_dir,
        )
        if isinstance(df_merged, tuple):
            df_feat = df_merged[0]
        else:
            df_feat = df_merged

        df_feat = df_feat.sort_values("date").reset_index(drop=True)

        # 3) data_cols 검증 및 채우기
        required = CFG_S.get("data_cols", [])
        if not required:
            raise ValueError(f"[SentimentalAgent.run_dataset] config에 data_cols가 정의되지 않았습니다.")
        
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        
        missing_after = [c for c in required if c not in df_feat.columns]
        if missing_after:
            print(f"[WARN] [{self.agent_id}] 누락된 feature {len(missing_after)}개를 0.0으로 채움: {missing_after[:5]}...")
            for col in missing_after:
                df_feat[col] = 0.0

        # 4) 입력 행렬 생성
        feat_values = df_feat[required].values.astype("float32")

        if len(feat_values) < self.window_size:
            raise ValueError(f"데이터 길이({len(feat_values)}) < 윈도우({self.window_size})")

        X_last = feat_values[-self.window_size:]
        X_last = X_last[None, :, :]  # (1, T, F)
        self._last_input = X_last

        # 5) StockData 생성
        last_row = df_feat.iloc[-1]
        last_price = float(last_row["close"])

        sd = StockData()
        sd.ticker = self.ticker
        sd.last_price = last_price
        sd.currency = "USD"
        sd.feature_cols = required
        sd.window_size = self.window_size
        sd.raw_df = df_feat

        sd.news_feats = {
            "news_count_7d": float(last_row.get("news_count_7d", 0)),
            "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
            "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
        }

        sd.snapshot = {
            "agent_id": self.agent_id,
            "feature_cols": sd.feature_cols,
            "window_size": sd.window_size,
            "news_feats": sd.news_feats,
            "raw_df": sd.raw_df,
        }

        sd.X_seq = X_last
        sd.SentimentalAgent = {
            "X_seq": X_last,
            "last_price": last_price,
        }

        self.stockdata = sd
        return sd

    # -------------------------------------------------------
    # 내부 헬퍼: _ensure_sentimental_csv
    # -------------------------------------------------------
    def _ensure_sentimental_csv(self, ticker: str, rebuild: bool = False) -> None:
        """
        SentimentalAgent 전용 CSV 생성 메서드 (뉴스 데이터 수집 및 병합)
        """
        raw_csv_path = os.path.join(os.path.dirname(self.data_dir), "raw", f"{ticker}_{self.agent_id}_raw.csv")
        
        if not rebuild and os.path.exists(raw_csv_path):
            return
        
        print(f"[{self.agent_id}] Raw CSV 생성 중...")
        
        # 기간 설정
        period_str = common_params.get("period", "2y")
        if period_str.endswith("y"):
            days = int(period_str[:-1]) * 365
        elif period_str.endswith("m"):
            days = int(period_str[:-1]) * 30
        elif period_str.endswith("d"):
            days = int(period_str[:-1])
        else:
            days = 2 * 365
        
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=days)

        # 가격 데이터 수집
        df_price = yf.download(ticker, start=start, end=end, auto_adjust=False)
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = [c[0].lower() for c in df_price.columns]
        else:
            df_price.columns = [c.lower() for c in df_price.columns]

        df_price = df_price.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })

        df_price["date"] = df_price.index
        df_price = df_price.reset_index(drop=True)

        # 뉴스 데이터 병합

        
        if not df_price.empty:
            if 'date' in df_price.columns:
                price_start_date = pd.to_datetime(df_price['date']).min()
            else:
                price_start_date = df_price.index.min()
        else:
            price_start_date = None

        df_news = update_news_db(ticker, base_dir=self.news_dir, target_start_date=price_start_date)
        
        if df_news.empty:
            print(f"[WARN] {ticker} 뉴스 데이터가 없습니다. 감성 피처를 0으로 채웁니다.")
            df_merged = df_price.copy()
            for col in ["news_count_7d", "sentiment_mean_7d", "sentiment_vol_7d", "news_count_1d", "sentiment_mean_1d"]:
                df_merged[col] = 0.0
        else:
            # 일별 집계
            daily_stats = df_news.groupby('date').agg(
                count=('sentiment_score', 'count'),
                mean=('sentiment_score', 'mean')
            )
            
            idx = pd.date_range(daily_stats.index.min(), daily_stats.index.max())
            daily_stats = daily_stats.reindex(idx, fill_value=0)
            daily_stats.index.name = 'date'
            
            daily_stats['news_count_1d'] = daily_stats['count']
            daily_stats['sentiment_mean_1d'] = daily_stats['mean']
            
            # Rolling 7d
            daily_stats['news_count_7d'] = daily_stats['count'].rolling(7).sum().fillna(0)
            daily_stats['sentiment_mean_7d'] = daily_stats['mean'].rolling(7).mean().fillna(0)
            daily_stats['sentiment_vol_7d'] = daily_stats['mean'].rolling(7).std().fillna(0)
            
            daily_stats = daily_stats.reset_index()
            
            # 병합
            df_price_copy = df_price.copy()
            if 'date' not in df_price_copy.columns:
                df_price_copy['date'] = df_price_copy.index
            
            df_price_copy['date'] = pd.to_datetime(df_price_copy['date'])
            if df_price_copy['date'].dt.tz is not None:
                df_price_copy['date'] = df_price_copy['date'].dt.tz_localize(None)
                
            if daily_stats['date'].dt.tz is not None:
                daily_stats['date'] = daily_stats['date'].dt.tz_localize(None)
            
            df_merged = pd.merge(df_price_copy, daily_stats, on='date', how='left')
            
            for col in ['news_count_1d', 'sentiment_mean_1d', 'news_count_7d', 'sentiment_mean_7d', 'sentiment_vol_7d']:
                if col in df_merged.columns:
                    df_merged[col] = df_merged[col].fillna(0)

        df_feat = df_merged.sort_values("date").reset_index(drop=True)

        # 피처 계산 (return_1d, hl_range, Volume)
        if "return_1d" not in df_feat.columns:
            df_feat["return_1d"] = df_feat["close"].pct_change().fillna(0)
        if "hl_range" not in df_feat.columns:
            df_feat["hl_range"] = ((df_feat["high"] - df_feat["low"]) /
                                   df_feat["close"].replace(0, np.nan)).fillna(0)
        if "Volume" not in df_feat.columns:
            df_feat["Volume"] = df_feat["volume"].fillna(0)
        for col in ["news_count_1d", "sentiment_mean_1d"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        # CSV 저장
        try:
            os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
            df_raw = df_feat.copy()
            if "date" in df_raw.columns:
                df_raw = df_raw.rename(columns={"date": "Date"})
            if "close" in df_raw.columns:
                df_raw = df_raw.rename(columns={"close": "Close"})

            # config에서 feature 목록 가져오기
            feature_cols = CFG_S.get("data_cols", [])
            if not feature_cols:
                raise ValueError(f"[{self.agent_id}] config에 data_cols가 정의되지 않았습니다.")
            
            cols_to_save = []
            if "Date" in df_raw.columns:
                cols_to_save.append("Date")
            for col in feature_cols:
                if col in df_raw.columns and col not in ("Date", "Close"):
                    cols_to_save.append(col)
            if "Close" in df_raw.columns:
                cols_to_save.append("Close")

            # 기간 필터링
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            end_date = pd.Timestamp.today().normalize()
            start_date = end - pd.Timedelta(days=days)
            
            df_raw = df_raw[df_raw["Date"] >= start_date].copy()
            df_raw = df_raw.sort_values("Date").reset_index(drop=True)
            df_raw["Date"] = df_raw["Date"].dt.strftime("%Y-%m-%d")

            df_raw[cols_to_save].to_csv(raw_csv_path, index=False)
            print(f"✅ [{self.agent_id}] Raw CSV 저장 완료: {raw_csv_path} ({len(df_raw):,} rows)")
        except Exception as e:
            print(f"❌ [{self.agent_id}] Raw CSV 저장 실패: {e}")

    # -------------------------------------------------------
    # searcher (통일된 CSV 기반 캐싱 패턴)
    # -------------------------------------------------------
    def search(self, ticker: Optional[str] = None, rebuild: bool = False):
        """
        SentimentalAgent 전용 Searcher
        - 데이터를 수집하고 최신 윈도우 텐서를 반환
        - 필요 시 CSV 파일 생성 (rebuild)
        """
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError(f"{agent_id}: ticker가 지정되지 않았습니다.")
        
        self.ticker = str(ticker).upper()
        
        raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
        raw_csv_path = os.path.join(raw_dir, f"{ticker}_{agent_id}_raw.csv")
        
        # 백테스팅 모드 처리
        if hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date:
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(temp_dir, f"{ticker}_{agent_id}_raw_{date_str}.csv")
            if os.path.exists(temp_path):
                raw_csv_path = temp_path
                print(f"[INFO] 백테스팅 모드: 필터링된 데이터셋 사용 ({self.simulation_date} 이전)")
        
        cfg = agents_info.get(agent_id, {})
        
        # Raw CSV 생성 (필요 시)
        if not (hasattr(self, 'test_mode') and self.test_mode and hasattr(self, 'simulation_date') and self.simulation_date and os.path.exists(raw_csv_path)):
            self._ensure_sentimental_csv(ticker, rebuild=rebuild)
        
        if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")
        
        # 데이터 로드 및 윈도우 추출
        df_raw = pd.read_csv(raw_csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)
        
        # config에서 feature 목록 가져오기
        feature_cols = cfg.get("data_cols", [])
        if not feature_cols:
            raise ValueError(f"[{agent_id}] config에 data_cols가 정의되지 않았습니다.")
        
        # 누락된 feature 확인 및 처리
        missing_cols = [col for col in feature_cols if col not in df_raw.columns]
        if missing_cols:
            print(f"[WARN] [{agent_id}] 누락된 feature {len(missing_cols)}개를 0.0으로 채움: {missing_cols[:5]}...")
            for col in missing_cols:
                df_raw[col] = 0.0
        
        window_size = cfg.get("window_size", self.window_size)
        
        X_all = df_raw[feature_cols].values.astype(np.float32)
        
        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우 크기({window_size})")
        
        x_latest = X_all[-window_size:].reshape(1, window_size, -1)  # (1, T, F)
        
        print(f"✅ [{agent_id}] Searcher 완료: 윈도우 shape {x_latest.shape}")
        
        # StockData 구성
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

        self.stockdata.X_seq = x_latest
        
        df_latest = pd.DataFrame(x_latest[0], columns=feature_cols)
        feature_dict = {col: df_latest[col].tolist() for col in df_latest.columns}
        setattr(self.stockdata, agent_id, feature_dict)
        
        if len(df_raw) > 0:
            last_row = df_raw.iloc[-1]
            self.stockdata.news_feats = {
                "news_count_7d": float(last_row.get("news_count_7d", 0)),
                "sentiment_mean_7d": float(last_row.get("sentiment_mean_7d", 0)),
                "sentiment_vol_7d": float(last_row.get("sentiment_vol_7d", 0)),
            }
            self.stockdata.raw_df = df_raw

        return torch.tensor(x_latest, dtype=torch.float32)

    # -------------------------------------------------------
    # predict
    # -------------------------------------------------------
    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None):
        """
        Monte Carlo Dropout을 이용한 예측 수행
        
        Returns:
            Target: 예측 결과 (종가, 불확실성, 신뢰도)
        """
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)
        
        # 입력 데이터 정리 (StockData -> Tensor)
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, "X_seq", None)
            if X_in is None:
                X_in = getattr(sd, self.agent_id, None)
                if isinstance(X_in, dict):
                    # StockData.feature_cols를 사용하여 순서 보장
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
        else:
            sd = None
            X_in = X

        if X_in is None:
            raise ValueError("predict() 입력 X가 None 입니다.")

        if isinstance(X_in, np.ndarray):
            X_raw_np = X_in.copy()
        elif isinstance(X_in, torch.Tensor):
            X_raw_np = X_in.detach().cpu().numpy().copy()
        else:
            raise TypeError(f"Unsupported input type for predict: {type(X_in)}")

        # 모델 준비
        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다.")
        
        # 모델 파일 체크 및 Pretrain
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
        
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError(f"{self.agent_id} 모델이 초기화되지 않음")

        # 스케일러 로드
        self.scaler.load(self.ticker)

        # 데이터 형태 맞춤
        if X_raw_np.ndim == 2:
            X_raw_np = X_raw_np[None, :, :]
        elif X_raw_np.ndim == 3 and X_raw_np.shape[0] != 1:
            raise ValueError(f"예상하지 못한 배치 크기: {X_raw_np.shape[0]}")
        
        X_scaled, _ = self.scaler.transform(X_raw_np)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        device = getattr(self, "device", torch.device("cpu"))
        X_tensor = X_tensor.to(device)
        model.to(device)

        # MC Dropout 추론
        model.train()
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                y_pred = model(X_tensor)
                if isinstance(y_pred, (tuple, list)):
                    y_pred = y_pred[0]
                preds.append(y_pred.detach().cpu().numpy().flatten())

        preds = np.stack(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))

        sigma = float(std_pred[-1])
        sigma_min = common_params.get("sigma_min", 1e-6)
        sigma = max(sigma, sigma_min)
        confidence = self._calculate_confidence_from_direction_accuracy()
        # 방향 정확도만 사용 (fallback 제거)

        # 역변환 및 가격 계산
        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        predicted_return = float(mean_pred[-1]) / y_scale_factor
        
        # 수익률 클리핑
        cfg = agents_info.get(self.agent_id, {})
        return_clip_min = cfg.get("return_clip_min", -0.5)
        return_clip_max = cfg.get("return_clip_max", 0.5)
        predicted_return = np.clip(predicted_return, return_clip_min, return_clip_max)

        if current_price is None:
            if sd is not None and getattr(sd, "last_price", None) is not None:
                current_price = float(sd.last_price)
            else:
                default_price = common_params.get("default_current_price", 100.0)
                current_price = float(getattr(self, "last_price", default_price))

        predicted_price = float(current_price * (1.0 + predicted_return))

        target = Target(
            next_close=predicted_price,
            uncertainty=sigma,
            confidence=confidence,
            predicted_return=float(predicted_return),
        )

        if hasattr(self, "targets"):
            self.targets.append(target)

        return target

    # -------------------------------------------------------
    # 내부 helper: _predict_next_close
    # -------------------------------------------------------
    @torch.inference_mode()
    def _predict_next_close(self) -> Tuple[float, float, float, List[str]]:
        """
        다음날 종가 예측 수행
        """
        sd = getattr(self, "stockdata", None)
        if sd is None or getattr(sd, "X_seq", None) is None:
            cfg = agents_info.get(self.agent_id, {})
            # 기간 설정
            period_str = common_params.get("period", "2y")
            if period_str.endswith("y"):
                days = int(period_str[:-1]) * 365
            else:
                days = 2 * 365
            
            sd = self.run_dataset(days=days)

        n_samples = common_params.get("n_samples", 30)
        target = self.predict(sd, n_samples=n_samples)
        cols = list(getattr(sd, "feature_cols", self.feature_cols))
        return float(target.next_close), float(target.uncertainty or 0.0), float(target.confidence or 0.0), cols

    # -------------------------------------------------------
    # ctx 구성 (프롬프트용)
    # -------------------------------------------------------
    def build_ctx(self, asof_date_kst: Optional[str] = None) -> Dict[str, Any]:
        """LLM 프롬프트용 Context 생성"""
        # 0) StockData 확인
        stockdata = getattr(self, "stockdata", None)
        if stockdata is None or getattr(stockdata, "X_seq", None) is None:
            stockdata = self.run_dataset()

        if asof_date_kst is None:
            asof_date_kst = datetime.now().strftime("%Y-%m-%d")

        # 예측 수행
        pred_close, uncertainty_std, confidence, cols = self._predict_next_close()

        # 가격 스냅샷
        price_snapshot = {}
        df = getattr(stockdata, "raw_df", None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            last = df.iloc[-1]
            price_snapshot["Close"] = float(last.get("close", np.nan))
            price_snapshot["Open"] = float(last.get("open", np.nan))
            price_snapshot["High"] = float(last.get("high", np.nan))
            price_snapshot["Low"] = float(last.get("low", np.nan))
            price_snapshot["Volume"] = float(last.get("volume", np.nan))
        else:
            price_snapshot = {
                "Close": getattr(stockdata, "last_price", np.nan),
                "Open": None, "High": None, "Low": None, "Volume": None,
            }

        # 뉴스 피처 정보
        nf = getattr(stockdata, "news_feats", {}) or {}
        
        sentiment_summary = {
            "mean_7d": float(nf.get("sentiment_mean_7d", 0.0)),
            "mean_30d": 0.0,
            "pos_ratio_7d": 0.0,
            "neg_ratio_7d": 0.0,
        }
        sentiment_vol = {"vol_7d": float(nf.get("sentiment_vol_7d", 0.0))}
        news_count = {"count_7d": int(nf.get("news_count_7d", 0))}
        has_news = bool(news_count["count_7d"] > 0)

        last_price = price_snapshot.get("Close", np.nan)
        pred_return = None
        if last_price and last_price == last_price:
            pred_return = float(pred_close / last_price - 1.0)

        snapshot = {
            "asof_date": asof_date_kst,
            "last_price": last_price,
            "currency": getattr(stockdata, "currency", "USD"),
            "window_size": self.window_size,
            "feature_cols_preview": [c for c in (cols or [])[:8]],
        }

        feature_importance = {
            "sentiment_score": sentiment_summary.get("mean_7d", 0.0),
            "sentiment_summary": sentiment_summary,
            "sentiment_volatility": sentiment_vol,
            "trend_7d": 0.0,
            "news_count": news_count,
            "has_news": has_news,
            "price_snapshot": price_snapshot,
        }

        ctx = {
            "agent_id": self.agent_id,
            "ticker": self.ticker,
            "snapshot": snapshot,
            "prediction": {
                "pred_close": pred_close,
                "pred_return": pred_return,
                "uncertainty": {
                    "std": uncertainty_std,
                    "ci95": float(1.96 * uncertainty_std),
                },
                "confidence": confidence,
                "pred_next_close": pred_close,
            },
            "feature_importance": feature_importance,
        }
        return ctx

    # -------------------------------------------------------
    # 프롬프트 빌더
    # -------------------------------------------------------
    def _build_messages_opinion(self, stock_data: StockData, target: Target) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        ctx = self.build_ctx()
        
        # 최근 뉴스 헤드라인 추가
        news_summary = []
        try:
            db_path = os.path.join(self.news_dir, f"{self.ticker}_news_db.csv")
            if os.path.exists(db_path):
                df_news = pd.read_csv(db_path)
                df_news['date'] = pd.to_datetime(df_news['date'])
                if df_news['date'].dt.tz is not None:
                    df_news['date'] = df_news['date'].dt.tz_localize(None)
                
                asof_date_str = ctx['snapshot']['asof_date']
                last_date = pd.to_datetime(asof_date_str)
                start_date = last_date - pd.Timedelta(days=7)
                
                recent_news = df_news[(df_news['date'] >= start_date) & (df_news['date'] <= last_date)]
                
                if not recent_news.empty:
                    recent_news = recent_news.copy()
                    recent_news['abs_score'] = recent_news['sentiment_score'].abs()
                    top_news = recent_news.sort_values('abs_score', ascending=False).head(5)
                    
                    for _, row in top_news.iterrows():
                        date_str = row['date'].strftime('%Y-%m-%d')
                        title = str(row['title'])
                        label = str(row['sentiment_label'])
                        score = float(row['sentiment_score'])
                        news_summary.append(f"- {date_str}: {title} ({label}, {score:.2f})")
        except Exception as e:
            print(f"[WARN] 뉴스 요약 생성 실패: {e}")
            
        ctx['recent_news_headlines'] = news_summary if news_summary else ["(최근 7일간 주요 뉴스 없음)"]

        # Target 정보 반영
        ctx["prediction"]["pred_next_close"] = float(getattr(target, "next_close", 0.0))
        ctx["prediction"]["pred_close"] = ctx["prediction"]["pred_next_close"]

        last_close = ctx["snapshot"].get("last_price")
        if isinstance(last_close, (int, float)) and last_close not in (0, None):
            try:
                chg = ctx["prediction"]["pred_next_close"] / float(last_close) - 1.0
            except ZeroDivisionError:
                chg = None
        else:
            chg = None
        ctx["prediction"]["pred_return"] = chg

        ctx_json = json.dumps(ctx, ensure_ascii=False, indent=2)
        prompts = OPINION_PROMPTS["SentimentalAgent"]
        system_text = prompts["system"]
        user_tmpl = prompts["user"]

        try:
            user_text = user_tmpl.format(context=ctx_json)
        except KeyError:
            user_text = user_tmpl.replace("{context}", ctx_json)

        return system_text, user_text

    def _build_messages_rebuttal(self, my_opinion: Opinion, target_opinion: Opinion, stock_data: StockData) -> Tuple[str, str]:
        opp_agent = getattr(target_opinion, "agent_id", "UnknownAgent")
        opp_reason = getattr(target_opinion, "reason", "")

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        system_tmpl = REBUTTAL_PROMPTS["SentimentalAgent"].get("system")
        user_tmpl = REBUTTAL_PROMPTS["SentimentalAgent"].get("user")

        user_text = user_tmpl.format(
            ticker=self.ticker,
            opp_agent=opp_agent,
            opp_reason=opp_reason if opp_reason else "(상대 의견 내용 없음)",
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=f"{sent.get('mean_7d', 0.0):.4f}",
            mean30=f"{sent.get('mean_30d', 0.0):.4f}",
            pos7=f"{sent.get('pos_ratio_7d', 0.0):.4f}",
            neg7=f"{sent.get('neg_ratio_7d', 0.0):.4f}",
            vol7=("NA" if vol7 is None else f"{vol7:.4f}"),
            trend7=("NA" if trend7 is None else f"{trend7:.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
        )
        return system_tmpl, user_text

    def _build_messages_revision(self, my_opinion: Opinion, others: List[Opinion], rebuttals: Optional[List[Rebuttal]] = None, stock_data: StockData = None) -> Tuple[str, str]:
        if stock_data is None:
            stock_data = self.stockdata

        def _op_text(x: Union[Opinion, Dict[str, Any], str, None, Any]) -> str:
            if isinstance(x, Opinion):
                return getattr(x, "reason", "")
            if isinstance(x, dict):
                return x.get("reason", "") or x.get("message", "")
            if hasattr(x, "message"):
                return getattr(x, "message", "")
            if hasattr(x, "reason"):
                return getattr(x, "reason", "")
            return str(x) if x else ""

        prev_reason = _op_text(my_opinion)
        reb_texts = []
        if isinstance(rebuttals, list):
            for r in rebuttals:
                reb_texts.append(_op_text(r))
        elif rebuttals is not None:
            reb_texts.append(_op_text(rebuttals))

        ctx = self.build_ctx()
        fi = ctx.get("feature_importance", {})
        sent = fi.get("sentiment_summary", {})
        vol7 = fi.get("sentiment_volatility", {}).get("vol_7d", None)
        trend7 = fi.get("trend_7d", None)
        news7 = fi.get("news_count", {}).get("count_7d", None)

        pred_close = float(my_opinion.target.next_close)
        last_price = ctx.get("snapshot", {}).get("last_price")
        change_ratio = None
        if last_price and last_price == last_price and last_price != 0:
            change_ratio = pred_close / last_price - 1.0

        # 컨텍스트 문장 생성 (상태 설명용)
        context_parts = []
        if last_price is not None:
            if change_ratio is not None:
                context_parts.append(f"현재 주가는 {last_price:.2f}이고, 다음 거래일 종가를 {pred_close:.2f}로 예측했습니다 ({change_ratio*100:.2f}%).")
            else:
                context_parts.append(f"현재 주가는 {last_price:.2f}이며, 예측값은 {pred_close:.2f}입니다.")
        
        mean7_val = sent.get('mean_7d', None)
        if mean7_val is not None:
            context_parts.append(f"최근 7일 평균 감성 점수는 {mean7_val:.3f}입니다.")
        if news7 is not None:
            context_parts.append(f"최근 7일 뉴스 개수는 {news7}건입니다.")

        context_str = " ".join(context_parts)

        system_tmpl = REVISION_PROMPTS["SentimentalAgent"].get("system")
        user_tmpl = REVISION_PROMPTS["SentimentalAgent"].get("user")

        rebuts_joined = "- " + "\n- ".join([s for s in reb_texts if s]) if reb_texts else "(반박 없음)"

        user_text = user_tmpl.format(
            ticker=self.ticker,
            prev=prev_reason if prev_reason else "(초안 없음)",
            rebuts=rebuts_joined,
            pred_close=f"{pred_close:.4f}",
            chg=("NA" if change_ratio is None else f"{change_ratio*100:.2f}%"),
            mean7=("NA" if mean7_val is None else f"{float(mean7_val):.4f}"),
            mean30="NA",
            pos7="NA",
            neg7="NA",
            vol7=("NA" if vol7 is None else f"{float(vol7):.4f}"),
            trend7=("NA" if trend7 is None else f"{float(trend7):.4f}"),
            news7=("NA" if news7 is None else f"{news7}"),
            context=context_str,
        )
        return system_tmpl, user_text

    def review_revise(self, my_opinion, others, rebuttals, stock_data=None):
        return super().review_revise(my_opinion, others, rebuttals, stock_data)

    def get_opinion(self, idx: int = 0, ticker: Optional[str] = None) -> Opinion:
        """단독 테스트용 Opinion 생성"""
        if ticker and ticker != self.ticker:
            self.ticker = str(ticker).upper()

        pred_close, uncertainty_std, confidence, _ = self._predict_next_close()
        target = Target(
            next_close=float(pred_close),
            uncertainty=float(uncertainty_std),
            confidence=float(confidence),
        )

        try:
            if hasattr(self, "reviewer_draft"):
                op = self.review_draft(getattr(self, "stockdata", None), target)
                return op
        except Exception as e:
            print("[SentimentalAgent] reviewer_draft 실패:", e)

        ctx = self.build_ctx()
        fi = ctx["feature_importance"]
        sent = fi["sentiment_summary"]

        reason = (
            f"{self.ticker}의 최근 7일 감성 평균은 {sent['mean_7d']:.3f}이며 "
            f"뉴스 개수는 {fi['news_count']['count_7d']}건입니다."
        )

        return Opinion(
            agent_id=self.agent_id,
            target=target,
            reason=reason,
        )
