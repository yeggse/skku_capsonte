# config/agents_set.py
# ===============================================================
# 에이전트 설정 및 하이퍼파라미터 관리
# ===============================================================
# 이 파일은 모든 에이전트(Technical, Macro, Sentimental)의 
# 학습 및 실행에 필요한 설정값을 정의합니다.
#
# 주요 구성:
# 1. common_params: 모든 에이전트가 공유하는 전역 설정
# 2. agents_info: 각 에이전트별 고유 하이퍼파라미터 및 모델 구조
# 3. dir_info: 데이터 및 모델 저장 경로
# ===============================================================

# 공통 파라미터 (모든 Agent에서 공유)
common_params = {
    # --- Monte Carlo Dropout 설정 ---
    "n_samples": 30,                    # 예측 시 샘플링 횟수 (불확실성 계산용)
    
    # --- LLM 설정 ---
    "temperature": 0.2,                 # LLM 생성 다양성 조절 (낮을수록 결정적)
    "preferred_models": ["gpt-5-mini", "gpt-4.1-mini"],  # 사용할 모델 우선순위 목록
    
    # --- 학습 및 Loss 설정 ---
    "huber_loss_delta": 1.0,            # Huber Loss의 delta 값 (이상치 민감도 조절)
    "fine_tune_lr": 1e-2,               # Fine-tuning 학습률
    "fine_tune_epochs": 10,             # Fine-tuning 에포크 수
    
    # --- 데이터 스케일링 및 처리 ---
    "y_scale_factor": 100.0,            # 수익률 타겟 스케일링 배수 (예: 0.01 -> 1.0)
    "eval_split_ratio": 0.8,            # 학습/검증 데이터 분할 비율 (0.8 = 80% 학습)
    "period": "2y",                     # 일반 모드 데이터 수집 기간
    "period_test": "2y",                # 백테스팅 모드 데이터 수집 기간
    "pretrain_save_dataset": True,      # 전처리된 데이터셋 저장 여부
    "pretrain_log_interval": 5,         # 학습 로그 출력 주기 (에포크 단위)
    
    # --- Early Stopping 설정 ---
    "early_stopping_enabled": True,     # Early Stopping 활성화 여부
    "early_stopping_min_delta": 1e-6,   # 최소 개선량 (original scale 기준)
    
    # --- 예측 및 불확실성 ---
    "default_current_price": 100.0,     # 현재가가 없을 경우 사용할 기본값
    "sigma_min": 1e-6,                  # 불확실성(표준편차) 최소값 (0 나누기 방지)
    "confidence_formula": "1.0 / (1.0 + sigma)",  # 신뢰도 계산 공식 (sigma가 클수록 신뢰도 하락)
    "direction_penalty_factor": 1.5,    # 방향성 오차에 대한 패널티 팩터 (1.0 = 패널티 없음)
    "confidence_lookback_days": 30,     # 신뢰도 계산에 사용할 최근 일수 (방향정확도 기반)
    
    # --- 앙상블 모델 (LightGBM) 하이퍼파라미터 ---
    "ensemble_n_estimators": 100,        # 트리 개수
    "ensemble_learning_rate": 0.05,     # 학습률
    "ensemble_max_depth": 3,            # 트리 최대 깊이
    "ensemble_random_state": 42,         # 랜덤 시드
    "ensemble_n_jobs": -1,              # 병렬 처리 (-1 = 모든 CPU 사용)
    "ensemble_verbosity": -1,           # 로그 출력 레벨 (-1 = 억제)
}

# 에이전트별 상세 설정
agents_info = {
    # -----------------------------------------------------------
    # 1. TechnicalAgent: 기술적 지표 기반 예측 (LSTM + Time-Attention)
    # -----------------------------------------------------------
    "TechnicalAgent": {
        "description": "기술적 지표(RSI, MACD 등)와 가격 데이터를 분석하여 예측",
        
        # 모델 아키텍처 정보 (참고용)
        "model_architecture": {
            "type": "LSTM_with_TimeAttention",
            "layers": [
                {"type": "LSTM", "input_dim": 13, "hidden_dim": 64, "name": "lstm1"},
                {"type": "LSTM", "input_dim": 64, "hidden_dim": 32, "name": "lstm2"},
                {"type": "TimeAttention", "hidden_dim": 32, "name": "attn_vec"},
                {"type": "Linear", "input_dim": 32, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 사용할 데이터 컬럼 (기술적 지표)
        "data_cols": [
            "weekofyear_sin", "weekofyear_cos", "log_ret_lag1",
            "ret_3d", "mom_10", "ma_200",
            "macd", "bbp", "adx_14",
            "obv", "vol_ma_20", "vol_chg", "vol_20d"
        ],
        "feature_builder": "core.technical_classes.technical:build_features_technical",
        
        # 모델 하이퍼파라미터
        "input_dim": 13,  
        "window_size": 20,  # 13w 최적화: AZN=20, CCEP=10, MSFT=20
        "rnn_units1": 64,  # 13w 최적화: AZN=128, CCEP=32, MSFT=64
        "rnn_units2": 32,  # 13w 최적화: AZN=16, CCEP=64, MSFT=32
        "dropout": 0.1,  # 13w 최적화: AZN=0.1, CCEP=0.1, MSFT=0.2
        "epochs": 100,  # 13w 최적화: AZN=45, CCEP=45, MSFT=45
        "patience": 20,  # 13w 최적화: AZN=20, CCEP=20, MSFT=20
        "learning_rate": 0.0042,  # 13w 최적화: AZN=0.0033, CCEP=0.0042, MSFT=4.247e-4
        "batch_size": 64,  # 13w 최적화: AZN=64, CCEP=32, MSFT=64
        "shuffle": False,  # 시계열 데이터 학습 시 셔플 여부 (True: 셔플, False: 시간 순서 유지)
        
        # 설정 및 기타
        "interval": "1d",               # 데이터 주기
        "x_scaler": "MinMaxScaler",     # 입력 데이터 스케일러
        "y_scaler": "StandardScaler",   # 타겟 데이터 스케일러
        "loss_fn": "HuberLoss",         # 손실 함수
        "seed": 1234,                   # 랜덤 시드
        
        # Debate 관련 파라미터
        "gamma": 0.3,                   # 의견 수렴율 (높을수록 타 에이전트 의견 수용도 높음)
        "delta_limit": 0.05,            # 최대 변화 허용 폭
        
        # TechnicalAgent 전용 설정
        "fine_tune_epochs": 20,         # Revise 단계 Fine-tuning 에포크
        "return_clip_min": -0.5,        # 수익률 클리핑 하한 (-50%)
        "return_clip_max": 0.5,         # 수익률 클리핑 상한 (+50%)
        
        # 설명가능성(XAI) 파라미터
        "occlusion_batch_size": 32,     # Occlusion 분석 배치 크기
        "top_k_features": 5,            # 주요 피처 추출 개수
        "shap_weight_time": 0.20,       # 시간 중요도 가중치 (SHAP)
        "shap_weight_feat": 0.30,       # 피처 중요도 가중치 (SHAP)
        "attention_weights": [0.4, 0.25, 0.15],  # 시간 중요도 융합 가중치 [Attn, Grad, Occ]
        "feature_weights": [0.5, 0.2],           # 피처 중요도 융합 가중치 [Grad, Occ]
        "pack_idea_top_time": 8,        # 설명 압축 시 포함할 상위 시간대 수
        "pack_idea_top_feat": 6,        # 설명 압축 시 포함할 상위 피처 수
        "pack_idea_coverage": 0.8,      # 설명 압축 커버리지 비율
    },

    # -----------------------------------------------------------
    # 2. MacroAgent: 거시경제 지표 기반 예측 (Stacked LSTM)
    # -----------------------------------------------------------
    "MacroAgent": {
        "description": "금리, 환율, 유가 등 거시경제 지표와 시장 심리 지수를 분석",
        
        # 모델 아키텍처 정보
        "model_architecture": {
            "type": "LSTM_Stacked_Dense",
            "layers": [
                {"type": "LSTM", "input_dim": "input_dim", "hidden_dim": 128, "name": "lstm1"},
                {"type": "LSTM", "input_dim": 128, "hidden_dim": 64, "name": "lstm2"},
                {"type": "LSTM", "input_dim": 64, "hidden_dim": 32, "name": "lstm3"},
                {"type": "Linear", "input_dim": 32, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 사용할 데이터 컬럼 (MACRO_TICKERS 기반 피처)
        "data_cols": [
            "CL=F_Close", "CL=F_High", "CL=F_Low", "CL=F_Open", "CL=F_Volume", "CL=F_ret_1d",
            "DX-Y.NYB_Close", "DX-Y.NYB_High", "DX-Y.NYB_Low", "DX-Y.NYB_Open", "DX-Y.NYB_Volume", "DX-Y.NYB_ret_1d",
            "EURUSD=X_Close", "EURUSD=X_High", "EURUSD=X_Low", "EURUSD=X_Open", "EURUSD=X_Volume", "EURUSD=X_ret_1d",
            "GC=F_Close", "GC=F_High", "GC=F_Low", "GC=F_Open", "GC=F_Volume", "GC=F_ret_1d",
            "HG=F_Close", "HG=F_High", "HG=F_Low", "HG=F_Open", "HG=F_Volume", "HG=F_ret_1d",
            "QQQ_Close", "QQQ_High", "QQQ_Low", "QQQ_Open", "QQQ_Volume", "QQQ_ret_1d",
            "Risk_Sentiment",
            "SPY_Close", "SPY_High", "SPY_Low", "SPY_Open", "SPY_Volume", "SPY_ret_1d",
            "USDJPY=X_Close", "USDJPY=X_High", "USDJPY=X_Low", "USDJPY=X_Open", "USDJPY=X_Volume", "USDJPY=X_ret_1d",
            "Yield_spread",
            "^DJI_Close", "^DJI_High", "^DJI_Low", "^DJI_Open", "^DJI_Volume", "^DJI_ret_1d",
            "^FVX_Close", "^FVX_High", "^FVX_Low", "^FVX_Open", "^FVX_Volume", "^FVX_ret_1d",
            "^GSPC_Close", "^GSPC_High", "^GSPC_Low", "^GSPC_Open", "^GSPC_Volume", "^GSPC_ret_1d",
            "^IRX_Close", "^IRX_High", "^IRX_Low", "^IRX_Open", "^IRX_Volume", "^IRX_ret_1d",
            "^IXIC_Close", "^IXIC_High", "^IXIC_Low", "^IXIC_Open", "^IXIC_Volume", "^IXIC_ret_1d",
            "^TNX_Close", "^TNX_High", "^TNX_Low", "^TNX_Open", "^TNX_Volume", "^TNX_ret_1d",
            "^VIX_Close", "^VIX_High", "^VIX_Low", "^VIX_Open", "^VIX_Volume", "^VIX_ret_1d",
            "ma10", "ma5", "ret1"
        ],
        
        # 모델 하이퍼파라미터
        "hidden_dims": [128,64,32],  # 13w 최적화: AZN=[128,64,32], CCEP=[128,64,32], MSFT=[128,64,32]
        "dropout_rates": [0.1, 0.1, 0.1],  # 13w 최적화: AZN=[0.1,0.1,0.1], CCEP=[0.1,0.1,0.1], MSFT=[0.1,0.1,0.1]
        "window_size": 40,  # 13w 최적화: AZN=40, CCEP=40, MSFT=60
        "epochs": 100,  # 13w 최적화: AZN=60, CCEP=60, MSFT=60
        "patience": 10,  # 13w 최적화: AZN=10, CCEP=10, MSFT=10
        "learning_rate": 0.005,  # 13w 최적화: AZN=0.005, CCEP=0.005, MSFT=0.005
        "batch_size": 32,  # 13w 최적화: AZN=32, CCEP=32, MSFT=16
        "shuffle": False,  # 시계열 데이터 학습 시 셔플 여부 (True: 셔플, False: 시간 순서 유지)
        
        # 설정 및 기타
        "interval": "1d",
        "x_scaler": "StandardScaler",  # 13w 최적화: AZN=StandardScaler, CCEP=RobustScaler, MSFT=StandardScaler
        "y_scaler": "MinMaxScaler",     # 타겟은 -1 ~ 1 범위로 스케일링
        "loss_fn": "HuberLoss",  # 13w 최적화: AZN=HuberLoss, CCEP=HuberLoss, MSFT=HuberLoss
        
        # Debate 관련 파라미터
        "gamma": 0.5,
        "delta_limit": 0.1,
        
        # MacroAgent 전용 설정
        "fine_tune_epochs": 5,
        "searcher_buffer_days": 50,     # 데이터 수집 시 여유 기간 (지표 계산용)
        "recent_days": 14,              # 최근 데이터 조회 기간
        "return_clip_min": -0.5,
        "return_clip_max": 0.5,
        "minmax_scaler_range": (-1, 1), # MinMaxScaler 범위
    },

    # -----------------------------------------------------------
    # 3. SentimentalAgent: 뉴스 및 감성 분석 (LSTM)
    # -----------------------------------------------------------
    "SentimentalAgent": {
        "description": "뉴스 헤드라인과 감성 점수, 거래량 변동 등을 분석",
        
        # 모델 아키텍처 정보
        "model_architecture": {
            "type": "SentimentalLSTM",
            "layers": [
                {"type": "LSTM", "input_dim": 8, "hidden_dim": 64, "num_layers": 2, "name": "lstm"},
                {"type": "Linear", "input_dim": 64, "output_dim": 1, "name": "fc"}
            ],
            "output": "next_day_return"
        },
        
        # 사용할 데이터 컬럼 (뉴스 및 감성 분석 피처)
        "data_cols": [
            "return_1d",
            "hl_range",
            "Volume",
            "news_count_1d",
            "news_count_7d",
            "sentiment_mean_1d",
            "sentiment_mean_7d",
            "sentiment_vol_7d"
        ],
        
        # 모델 하이퍼파라미터
        "input_dim": 8,
        "d_model": 64,  # 13w 최적화: AZN=64, CCEP=96, MSFT=96
        "nhead": 4,                     # (참고용) Attention 헤드 수
        "num_layers": 2,                # LSTM 층 수
        "dropout": 0.2,  # 13w 최적화: AZN=0.2, CCEP=0.2, MSFT=0.2
        "window_size": 15,  # 13w 최적화: AZN=20, CCEP=15, MSFT=15
        "epochs": 100,  # 13w 최적화: AZN=50, CCEP=50, MSFT=50
        "patience": 20,  # Early Stopping patience
        "learning_rate": 0.0003,  # 13w 최적화: AZN=0.0001, CCEP=0.0005, MSFT=0.0003
        "batch_size": 32,  # 13w 최적화: AZN=32, CCEP=32, MSFT=32
        "shuffle": False,  # 시계열 데이터 학습 시 셔플 여부 (True: 셔플, False: 시간 순서 유지)
        
        # 설정 및 기타
        "interval": "1d",
        "x_scaler": "StandardScaler",
        "y_scaler": "StandardScaler",
        "loss_fn": "HuberLoss",
        
        # Debate 관련 파라미터
        "gamma": 0.3,
        "delta_limit": 0.05,
        
        # SentimentalAgent 전용 설정
        "return_clip_min": -0.5,
        "return_clip_max": 0.5,
    },
}

# 디렉토리 설정
dir_info = {
    "data_dir": "data/processed",       # 전처리된 데이터 저장 경로
    "model_dir": "models",              # 학습된 모델 저장 경로
    "scaler_dir": "models/scalers",     # 스케일러 저장 경로
    "artifacts_dir": "artifacts"        # 기타 결과물 저장 경로
}
