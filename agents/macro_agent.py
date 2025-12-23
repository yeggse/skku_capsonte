# agents/macro_agent.py

import os
import json
from dataclasses import asdict
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import TensorDataset, DataLoader

from config.agents_set import dir_info, agents_info, common_params
from core.macro_classes.macro_csv_builder import MacroCSVBuilder
from core.macro_classes.macro_llm import GradientAnalyzer
from agents.base_agent import BaseAgent, Target, StockData, Opinion, Rebuttal
from config.prompts import OPINION_PROMPTS, REBUTTAL_PROMPTS, REVISION_PROMPTS

class MacroAgent(BaseAgent, nn.Module):
    """거시경제 데이터를 기반으로 주가를 예측하는 에이전트"""

    def __init__(self,
                 base_date=datetime.today(),
                 window=None,
                 ticker=None,
                 agent_id='MacroAgent',
                 data_dir=None,
                 model_dir=None,
                 **kwargs):
        """MacroAgent 초기화"""
        nn.Module.__init__(self)

        if data_dir is None:
            data_dir = dir_info.get("data_dir", "data/processed")
        if model_dir is None:
            model_dir = dir_info.get("model_dir", "models")
        BaseAgent.__init__(self, agent_id=agent_id, ticker=ticker, data_dir=data_dir, model_dir=model_dir, **kwargs)

        cfg = agents_info.get(agent_id, {})

        self.agent_id = agent_id
        self.base_date = base_date
        self.window = int(window) if window is not None else cfg.get("window_size", 40)
        self.window_size = self.window
        self.tickers = [ticker] if ticker else []
        self.ticker = ticker

        if ticker:
            self.model_path = os.path.join(self.model_dir, f"{ticker}_{agent_id}.pt")
            scaler_dir = os.path.join(self.model_dir, "scalers")
            self.scaler_X_path = os.path.join(scaler_dir, f"{ticker}_{agent_id}_xscaler.pkl")
            self.scaler_y_path = os.path.join(scaler_dir, f"{ticker}_{agent_id}_yscaler.pkl")
        else:
            self.model_path = None
            self.scaler_X_path = None
            self.scaler_y_path = None

        data_cols = cfg.get("data_cols", [])
        self.input_dim = cfg.get("input_dim", len(data_cols) if data_cols else 95)
        self.output_dim = len(self.tickers) if self.tickers else 1
        hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
        dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

        self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)

        self.drop1 = nn.Dropout(dropout_rates[0])
        self.drop2 = nn.Dropout(dropout_rates[1])
        self.drop3 = nn.Dropout(dropout_rates[2])

        self.fc1 = nn.Linear(hidden_dims[2], 32)
        self.fc2 = nn.Linear(32, self.output_dim)

        self.scaler_X = None
        self.scaler_y = None
        self.macro_df = None
        self.X_scaled = None
        self.X_raw = None
        self.last_price = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_builder = MacroCSVBuilder(self.agent_id, self.data_dir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """모델 Forward Pass"""
        h1, _ = self.lstm1(x)
        h1 = self.drop1(h1)
        h2, _ = self.lstm2(h1)
        h2 = self.drop2(h2)
        h3, _ = self.lstm3(h2)
        h3 = self.drop3(h3)
        h3_last = h3[:, -1, :]
        out = torch.relu(self.fc1(h3_last))
        out = self.fc2(out)
        return out

    def _get_feature_cols(self):
        """피처 컬럼 리스트를 반환합니다"""
        cfg = agents_info.get(self.agent_id, {})
        return cfg.get("data_cols", [])


    def search(self, ticker: Optional[str] = None, rebuild: bool = False):
        """데이터를 검색하고 최신 윈도우 텐서를 반환합니다"""
        agent_id = self.agent_id
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError(f"{agent_id}: ticker가 지정되지 않았습니다.")

        self.ticker = ticker
        if ticker not in self.tickers:
            self.tickers = [ticker]

        # 모델 / 스케일러 경로 설정 (유지해도 됨)
        self.model_path = os.path.join(self.model_dir, f"{ticker}_{agent_id}.pt")
        scaler_dir = os.path.join(self.model_dir, "scalers")
        self.scaler_X_path = os.path.join(scaler_dir, f"{ticker}_{agent_id}_xscaler.pkl")
        self.scaler_y_path = os.path.join(scaler_dir, f"{ticker}_{agent_id}_yscaler.pkl")

        # ----------------------------
        # CSV 확보 (단일 책임)
        # ----------------------------
        csv_path = self.csv_builder.ensure_csv(ticker, rebuild=rebuild)

        # 백테스트 override
        if getattr(self, "test_mode", False) and getattr(self, "simulation_date", None):
            raw_dir = os.path.join(os.path.dirname(self.data_dir), "raw")
            temp_dir = os.path.join(raw_dir, "backtest_temp")
            date_str = self.simulation_date.replace("-", "")
            temp_path = os.path.join(
                temp_dir, f"{ticker}_{self.agent_id}_raw_{date_str}.csv"
            )
            if os.path.exists(temp_path):
                csv_path = temp_path
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용")

        # ----------------------------
        # CSV 로드
        # ----------------------------
        df_raw = pd.read_csv(csv_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw = df_raw.sort_values("Date").reset_index(drop=True)

        cfg = agents_info.get(agent_id, {})
        feature_cols = cfg.get("data_cols", [])
        if not feature_cols:
            raise ValueError(f"[{agent_id}] config에 data_cols가 정의되지 않았습니다.")

        # 누락 feature 보정
        missing_cols = [c for c in feature_cols if c not in df_raw.columns]
        if missing_cols:
            print(f"[WARN] [{agent_id}] 누락된 feature {len(missing_cols)}개 → 0.0 보정")
            for c in missing_cols:
                df_raw[c] = 0.0

        window_size = self.window
        X_all = df_raw[feature_cols].values.astype(np.float32)

        if len(X_all) < window_size:
            raise ValueError(f"데이터 길이({len(X_all)}) < 윈도우({window_size})")

        self.X_raw = df_raw[feature_cols]
        x_latest = X_all[-window_size:].reshape(1, window_size, -1)

        print(f"✅ [{agent_id}] Searcher 완료: 윈도우 shape {x_latest.shape}")

        # ----------------------------
        # StockData 구성
        # ----------------------------
        self.stockdata = StockData(ticker=ticker)
        self.stockdata.feature_cols = feature_cols
        self.stockdata.window_size = window_size

        try:
            self.stockdata.last_price = float(df_raw["Close"].iloc[-1])
            self.last_price = self.stockdata.last_price
        except Exception:
            self.stockdata.last_price = None

        try:
            self.stockdata.currency = yf.Ticker(ticker).info.get("currency", "USD")
        except Exception:
            self.stockdata.currency = "USD"

        df_latest = pd.DataFrame(x_latest[0], columns=feature_cols)
        setattr(
            self.stockdata,
            agent_id,
            {c: df_latest[c].tolist() for c in feature_cols},
        )

        return torch.tensor(x_latest, dtype=torch.float32)

    def pretrain(self):
        """모델을 사전학습합니다"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pretraining {self.agent_id}")

        cfg = agents_info.get(self.agent_id, {})
        epochs = cfg.get("epochs", 60)
        lr = cfg.get("learning_rate", 0.0005)
        batch_size = cfg.get("batch_size", 16)

        if not self.ticker:
            raise ValueError("MacroAgent.pretrain: ticker가 설정되지 않았습니다.")

        ticker = self.ticker

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
                print(f"[INFO] 백테스팅 모드: {self.simulation_date} 이전 데이터 사용 중, 마지막 타겟 제거 (데이터 누수 방지)")

        window_size = self.window
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

        if len(X_seq) == 0:
            print("[WARN] MacroAgent.pretrain: 학습용 시퀀스가 없습니다.")
            return

        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        y_seq = y_seq * y_scale_factor

        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(ticker)

        X_train, y_train = map(torch.tensor, self.scaler.transform(X_seq, y_seq))
        X_train, y_train = X_train.float(), y_train.float()

        actual_input_dim = X_seq.shape[-1]
        if actual_input_dim != self.input_dim:
            print(f"[INFO] input_dim 조정: {self.input_dim} -> {actual_input_dim}")
            self.input_dim = actual_input_dim
            hidden_dims = cfg.get("hidden_dims", [128, 64, 32])
            dropout_rates = cfg.get("dropout_rates", [0.3, 0.3, 0.2])

            self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
            self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
            self.lstm3 = nn.LSTM(hidden_dims[1], hidden_dims[2], batch_first=True)
            self.fc1 = nn.Linear(hidden_dims[2], 32)
            self.fc2 = nn.Linear(32, self.output_dim)

        model = self
        model.to(self.device)
        model.train()

        shuffle = cfg.get("shuffle", True)
        early_stopping_enabled = common_params.get("early_stopping_enabled", True)
        patience = cfg.get("patience", 10)
        min_delta = common_params.get("early_stopping_min_delta", 1e-6)
        eval_split_ratio = common_params.get("eval_split_ratio", 0.8)
        
        if early_stopping_enabled:
            split_idx = int(len(X_train) * eval_split_ratio)
            X_train_split = X_train[:split_idx]
            X_val_split = X_train[split_idx:]
            y_train_split = y_train[:split_idx]
            y_val_split = y_train[split_idx:]
            
            val_dataset = TensorDataset(X_val_split.to(self.device), y_val_split.to(self.device))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            X_train_split = X_train
            y_train_split = y_train
        
        train_dataset = TensorDataset(X_train_split.to(self.device), y_train_split.to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loss_fn_name = cfg.get("loss_fn", "L1Loss")
        if loss_fn_name == "HuberLoss":
            huber_delta = common_params.get("huber_loss_delta", 1.0)
            loss_fn = nn.HuberLoss(delta=huber_delta)
        elif loss_fn_name == "L1Loss":
            loss_fn = nn.L1Loss()
        elif loss_fn_name == "MSELoss":
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.L1Loss()

        log_interval = common_params.get("pretrain_log_interval", 5)
        final_loss = None
        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        best_val_loss_orig = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_loss_original = 0.0
            count = 0
            
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = loss_fn(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                with torch.no_grad():
                    pred_np = pred.detach().cpu().numpy()
                    by_np = by.detach().cpu().numpy()
                    pred_scaled = self.scaler.inverse_y(pred_np)
                    true_scaled = self.scaler.inverse_y(by_np)
                    pred_orig = pred_scaled / y_scale_factor
                    true_orig = true_scaled / y_scale_factor
                    mse_orig = np.mean((pred_orig - true_orig) ** 2)
                    train_loss_original += mse_orig
                    count += 1

            train_loss /= max(len(train_loader), 1)
            train_loss_original /= max(count, 1)
            final_loss = train_loss
            
            val_loss_orig = None
            if early_stopping_enabled:
                model.eval()
                val_loss_orig = 0.0
                val_count = 0
                
                with torch.no_grad():
                    for bx, by in val_loader:
                        pred = model(bx)
                        pred_np = pred.cpu().numpy()
                        by_np = by.cpu().numpy()
                        pred_scaled = self.scaler.inverse_y(pred_np)
                        true_scaled = self.scaler.inverse_y(by_np)
                        pred_orig = pred_scaled / y_scale_factor
                        true_orig = true_scaled / y_scale_factor
                        mse_orig = np.mean((pred_orig - true_orig) ** 2)
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
                    print(f"  Epoch {epoch+1:03d}/{epochs} | Loss (scaled): {train_loss:.6f} | Loss (original): {train_loss_original:.6f} | Val Loss (original): {val_loss_orig:.6f}")
                else:
                    print(f"  Epoch {epoch+1:03d}/{epochs} | Loss (scaled): {train_loss:.6f} | Loss (original): {train_loss_original:.6f}")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, self.model_path)
        self.model_loaded = True

        final_loss_str = f" (Final Loss: {final_loss:.6f})" if final_loss is not None else ""
        print(f"✅ {self.agent_id} 모델 학습 및 저장 완료: {self.model_path}{final_loss_str}")

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

    def load_model(self, model_path: Optional[str] = None):
        """저장된 모델 가중치를 로드합니다"""
        if model_path is None:
            if hasattr(self, "model_path") and self.model_path:
                model_path = self.model_path
            else:
                model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            if "lstm1.weight_ih_l0" in state_dict:
                weight = state_dict["lstm1.weight_ih_l0"]
                saved_input_dim = weight.shape[1]

                if saved_input_dim != self.input_dim:
                    print(f"[INFO] 모델 로드 중 input_dim 조정: {self.input_dim} -> {saved_input_dim}")
                    self.input_dim = saved_input_dim

                    cfg = agents_info.get(self.agent_id, {})
                    hidden_dims = cfg.get("hidden_dims", [128, 64, 32])

                    self.lstm1 = nn.LSTM(self.input_dim, hidden_dims[0], batch_first=True)
                    self.to(self.device)

            self.load_state_dict(state_dict, strict=False)
            self.eval()
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"[{self.agent_id}] load_model 실패: {e}")
            return False



    def review_draft(self, stock_data: StockData = None, target: Target = None) -> Opinion:
        """초기 의견을 생성합니다"""
        if stock_data is None:
            stock_data = self.stockdata
        if stock_data is None:
            if not self.ticker:
                raise ValueError("ticker가 설정되지 않았습니다.")
            self.search(self.ticker)
            stock_data = self.stockdata

        if target is None:
            X_input = self.search(self.ticker)
            target = self.predict(X_input)

        if self.X_raw is not None:
            try:
                self.scaler.load(self.ticker)

                X_window = self.X_raw.tail(self.window).values
                if X_window.ndim == 2:
                    X_window = X_window[None, :, :]

                X_scaled, _ = self.scaler.transform(X_window)
                X_scaled_np = X_scaled.astype(np.float32)

                cfg = agents_info.get(self.agent_id, {})
                feature_names = cfg.get("data_cols", [])
                if X_scaled_np.shape[2] > 300:
                    X_scaled_np = X_scaled_np[:, :, :300]
                    feature_names = feature_names[:300]

                gradient_analyzer = GradientAnalyzer(self, feature_names)
                importance_dict, temporal_df, consistency_df, sensitivity_df, grad_results = gradient_analyzer.run_all_gradients(X_scaled_np)

                if stock_data:
                    feature_imp = {
                        'feature_summary': grad_results.get("feature_summary"),
                        'importance_dict': importance_dict,
                        'temporal_summary': temporal_df.head().to_dict(orient="records") if temporal_df is not None else [],
                        'consistency_summary': consistency_df.to_dict(orient="records") if consistency_df is not None else [],
                        'sensitivity_summary': sensitivity_df.to_dict(orient="records") if sensitivity_df is not None else [],
                        'stability_summary': grad_results.get("stability_summary")
                    }

                    agent_data = getattr(stock_data, self.agent_id, {})
                    if isinstance(agent_data, dict):
                        agent_data['feature_importance'] = feature_imp
                        agent_data['our_prediction'] = target.next_close
                        agent_data['uncertainty'] = round(target.uncertainty or 0.0, 8)
                        agent_data['confidence'] = round(target.confidence or 0.0, 8)
                        setattr(stock_data, self.agent_id, agent_data)
            except Exception as e:
                print(f"[WARN] GradientAnalyzer 실행 실패: {e}")

        sys_text, user_text = self._build_messages_opinion(stock_data, target)
        parsed = self._ask_with_fallback(
            self._msg("system", sys_text),
            self._msg("user", user_text),
            {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"], "additionalProperties": False}
        )
        reason = parsed.get("reason", "(사유 생성 실패)")

        opinion = Opinion(agent_id=self.agent_id, target=target, reason=reason)
        self.opinions.append(opinion)
        return opinion

    def review_rebut(self, my_opinion: Opinion, other_opinion: Opinion, round: int) -> Rebuttal:
        """상대 의견에 대한 반박을 생성합니다"""
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
            message=parsed.get("message", "(실패)"),
            support_rate=support_rate
        )
        self.rebuttals[round].append(result)
        return result

    def review_revise(self, my_opinion, others, rebuttals, stock_data, fine_tune=True, lr: Optional[float] = None, epochs: Optional[int] = None):
        """의견을 수정합니다"""
        if lr is None:
            lr = common_params.get("fine_tune_lr", 1e-4)
        if epochs is None:
            epochs = agents_info.get(self.agent_id, {}).get("fine_tune_epochs", 5)
        return super().review_revise(my_opinion, others, rebuttals, stock_data, fine_tune, lr, epochs)

    def _build_messages_opinion(self, stock_data, target):
        """Opinion 생성 프롬프트를 구성합니다"""
        agent_data = getattr(stock_data, self.agent_id, None) or {}
        stock_data_dict = asdict(stock_data)
        feature_imp = agent_data.get("feature_importance", {})

        ctx = {
            "agent_id": self.agent_id,
            "ticker": stock_data_dict.get("ticker", "Unknown"),
            "currency": stock_data_dict.get("currency", "USD"),
            "last_price": stock_data_dict.get("last_price", None),
            "our_prediction": float(target.next_close),
            "uncertainty": float(target.uncertainty or 0.0),
            "confidence": float(target.confidence or 0.0),

            "feature_importance": {
                "feature_summary": feature_imp.get("feature_summary", []),
                "importance_dict": feature_imp.get("importance_dict", []),
                "temporal_summary": feature_imp.get("temporal_summary", []),
                'consistency_summary': feature_imp.get('consistency_summary', []),
                'sensitivity_summary': feature_imp.get('sensitivity_summary', []),
                'stability_summary': feature_imp.get('stability_summary', [])
            },
        }

        cfg = agents_info.get(self.agent_id, {})
        recent_days = cfg.get("recent_days", 14)
        for col, values in agent_data.items():
            if col == 'feature_importance': continue
            if isinstance(values, (list, tuple)):
                ctx[col] = values[-recent_days:]
            else:
                ctx[col] = [values]

        system_text = OPINION_PROMPTS[self.agent_id]["system"]
        user_text = OPINION_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_rebuttal(self, my_opinion, target_opinion, stock_data):
        """Rebuttal 생성 프롬프트를 구성합니다"""
        t = getattr(stock_data, "ticker", "UNKNOWN")
        ccy = getattr(stock_data, "currency", "USD")

        ctx = {
            "ticker": t,
            "currency": ccy,
            "me": {
                "agent_id": self.agent_id,
                "next_close": float(my_opinion.target.next_close),
                "reason": my_opinion.reason
            },
            "other": {
                "agent_id": target_opinion.agent_id,
                "next_close": float(target_opinion.target.next_close),
                "reason": target_opinion.reason
            }
        }

        system_text = REBUTTAL_PROMPTS[self.agent_id]["system"]
        user_text = REBUTTAL_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text

    def _build_messages_revision(self, my_opinion, others, rebuttals, stock_data):
        """Revision 생성 프롬프트를 구성합니다"""
        t = getattr(stock_data, "ticker", "UNKNOWN")
        others_summary = [{"agent": o.agent_id, "price": o.target.next_close, "reason": o.reason} for o in others]

        ctx = {
            "ticker": t,
            "my_opinion": {"price": my_opinion.target.next_close, "reason": my_opinion.reason},
            "others": others_summary,
            "rebuttals": [r.message for r in (rebuttals or [])]
        }

        system_text = REVISION_PROMPTS[self.agent_id]["system"]
        user_text = REVISION_PROMPTS[self.agent_id]["user"].format(context=json.dumps(ctx, ensure_ascii=False))
        return system_text, user_text






    def predict(self, X, n_samples: Optional[int] = None, current_price: Optional[float] = None,):
        if n_samples is None:
            n_samples = common_params.get("n_samples", 30)

        if not self.ticker:
            raise ValueError("ticker가 설정되지 않았습니다. search()를 먼저 호출하세요.")

        # 1. 모델 & 스케일러 보장
        self._ensure_model_and_scaler()

        # 2. 입력 정규화
        X_raw_np, current_price = self._prepare_input(X, current_price)

        # 3. 스케일링
        X_tensor = self._scale_input(X_raw_np)

        # 4. MC Dropout
        preds = self._run_mc_dropout(X_tensor, n_samples)

        # 5. 후처리
        mean_pred, std_pred = self._postprocess_prediction(preds)

        # 6. Target 생성
        return self._build_target(mean_pred, std_pred, current_price)
    def _ensure_model_and_scaler(self):
        model_path = os.path.join(self.model_dir, f"{self.ticker}_{self.agent_id}.pt")

        if not os.path.exists(model_path):
            if not getattr(self, "_in_pretrain", False):
                self._in_pretrain = True
                try:
                    self.pretrain()
                finally:
                    self._in_pretrain = False
            else:
                raise RuntimeError("pretrain 중 predict 재귀 호출")
        else:
            if not getattr(self, "model_loaded", False):
                self.load_model()

        self.scaler.load(self.ticker)
    def _prepare_input(self, X, current_price: Optional[float]):
        if isinstance(X, StockData):
            sd = X
            X_in = getattr(sd, self.agent_id, None)

            if isinstance(X_in, dict):
                feature_cols = sd.feature_cols
                df = pd.DataFrame({c: X_in.get(c, []) for c in feature_cols})
                X_in = df.values

            if current_price is None:
                current_price = sd.last_price

            X = X_in

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif not isinstance(X, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(X)}")

        if X.ndim == 2:
            X = X[None, :, :]

        if current_price is None:
            default_price = common_params.get("default_current_price", 100.0)
            current_price = default_price

        return X, float(current_price)
    def _scale_input(self, X_raw_np: np.ndarray) -> torch.Tensor:
        X_scaled, _ = self.scaler.transform(X_raw_np)
        device = self.device
        return torch.tensor(X_scaled, dtype=torch.float32).to(device)
    def _run_mc_dropout(self, X_tensor: torch.Tensor, n_samples: int) -> np.ndarray:
        self.train()  # dropout 활성화
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self(X_tensor)
                preds.append(out.cpu().numpy().flatten())

        return np.stack(preds)
    def _postprocess_prediction(self, preds: np.ndarray):
        mean_pred = preds.mean(axis=0)
        std_pred = np.abs(preds.std(axis=0))

        if hasattr(self.scaler, "y_scaler") and self.scaler.y_scaler is not None:
            mean_pred = self.scaler.inverse_y(mean_pred)
            std_pred = self.scaler.inverse_y(std_pred)

        return mean_pred, std_pred
    def _build_target(self, mean_pred: np.ndarray, std_pred: np.ndarray, current_price: float,) -> Target:
        sigma = float(std_pred[-1])
        sigma = max(sigma, common_params.get("sigma_min", 1e-6))

        confidence = self._calculate_confidence_from_direction_accuracy()
        if confidence is None:
            confidence = 0.5

        y_scale_factor = common_params.get("y_scale_factor", 100.0)
        predicted_return = float(mean_pred[-1]) / y_scale_factor

        cfg = agents_info.get(self.agent_id, {})
        predicted_return = np.clip(
            predicted_return,
            cfg.get("return_clip_min", -0.5),
            cfg.get("return_clip_max", 0.5),
        )

        predicted_price = current_price * (1.0 + predicted_return)

        return Target(
            next_close=float(predicted_price),
            uncertainty=sigma,
            confidence=confidence,
            predicted_return=float(predicted_return),
        )
