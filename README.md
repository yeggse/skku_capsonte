# 🧠 AI Stock Debate System (MVP)

> **신뢰도 기반 Revise를 수행하는 AI 주식 토론 시스템**

## 📋 프로젝트 개요

여러 전문 AI 에이전트가 주식에 대해 토론하고, 신뢰도 기반으로 의견을 수정하여 최종 예측을 도출하는 시스템입니다.

### 🎯 핵심 기능
- **다중 전문 에이전트**: Technical, Sentimental, MacroSenti 분석가
- **신뢰도 기반 Revise**: 각 에이전트의 불확실성(σ)을 기반으로 가중치 계산
- **실시간 대시보드**: Streamlit을 통한 인터랙티브 시각화
- **실시간 주가 연동**: Yahoo Finance API를 통한 현재가 정보

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TechnicalAgent  │    │ SentimentalAgent│    │ MacroSentiAgent │
│   (공격적)      │    │   (중립적)      │    │   (거시경제)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   DebateSystem   │
                    │  (토론 관리)    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Streamlit 대시보드│
                    │  (시각화)       │
                    └─────────────────┘
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:
```bash
CAPSTONE_OPENAI_API=your_openai_api_key_here
EODHD_API_KEY=your_eodhd_api_key_here  # SentimentalAgent 사용 시
```

### 3. 실행 방법

#### 방법 1: Streamlit 대시보드 (권장)
```bash
streamlit run streamlit_dashboard.py
```
대시보드에서:
1. 사이드바에서 종목 티커 입력 (예: AAPL, TSLA, NVDA)
2. 라운드 수 설정 (1-5)
3. "🚀 토론 시작" 버튼 클릭

#### 방법 2: CLI (Command Line Interface)
```bash
# 기본 실행 (3라운드)
python main.py --ticker NVDA

# 라운드 수 지정
python main.py --ticker TSLA --rounds 5

# 강제 pretrain 포함 (데이터셋 재생성 및 모델 재학습)
python main.py --ticker AAPL --rounds 3 --force-pretrain

# 데이터/모델 디렉토리 지정
python main.py --ticker MSFT --data-dir custom_data --model-dir custom_models
```

## 🎭 에이전트 소개

### 📈 TechnicalAgent (기술적 분석가)
- **특징**: 공격적, 차트 패턴 분석
- **예측 범위**: ±15%
- **분석 요소**: 이동평균, RSI, MACD, 볼린저밴드, OBV, ADX 등
- **모델 아키텍처**: 
  - LSTM with Time-Attention (2-layer LSTM: 64→32 units)
  - 입력 차원: 13개 기술 지표
  - 윈도우 크기: 20일 (기본값)
  - 설명가능성: Grad×Input, Occlusion, Attention 융합 분석
- **주요 기능**: 
  - 시점별/피처별 중요도 분석
  - SHAP 기반 Feature Importance
  - Fine-tuning을 통한 예측 보정

### 💭 SentimentalAgent (센티멘탈 분석가)
- **특징**: 중립적, 시장 심리 분석
- **예측 범위**: ±10%
- **분석 요소**: 뉴스 헤드라인, FinBERT 감성 점수, 거래량 변동 등
- **모델 아키텍처**:
  - SentimentalLSTM (2-layer LSTM: 64 units)
  - 입력 차원: 8개 피처 (수익률, 뉴스 개수, 감성 평균/변동성 등)
  - 윈도우 크기: 15일 (기본값)
  - Monte Carlo Dropout: 불확실성 추정 (n_samples=30)
- **주요 기능**:
  - EODHD API를 통한 뉴스 데이터 수집
  - FinBERT 기반 감성 분석
  - 7일/30일 감성 추세 비교

### 📊 MacroAgent (매크로 센티멘탈 분석가)
- **특징**: 거시경제 지표 기반 분석
- **예측 범위**: ±12%
- **분석 요소**: SPY, QQQ, VIX, 금리(IRX, TNX, FVX), 환율(EUR/USD, USD/JPY), 원자재(금, 구리, 원유) 등
- **모델 아키텍처**:
  - Stacked LSTM (3-layer: 128→64→32 units)
  - 입력 차원: 50+ 거시경제 변수
  - 윈도우 크기: 40일 (기본값)
  - 설명가능성: Gradient × Input, Integrated Gradients
- **주요 기능**:
  - 다중 자산 시계열 예측
  - 변수 중요도 및 일관성 분석
  - 민감도 및 안정성 평가

## 🔄 토론 프로세스

### Round 0: 초기 의견 생성
각 에이전트가 독립적으로 주식 분석 및 예측 수행

### Round 1-N: 토론 및 수정
1. **Rebuttal**: 다른 에이전트의 의견에 대한 반박/지지
2. **Revise**: 신뢰도 기반으로 자신의 의견 수정
3. **Ensemble**: 최종 예측가 계산

### 신뢰도 계산 공식
```
β_i = (1/σ_i) / Σ(1/σ_j)
revised_price = β_i × my_price + (1-β_i) × weighted_others
```

## 📊 대시보드 기능

### 🎯 주요 탭
- **최종의견 표**: 각 에이전트의 최종 예측가와 근거
- **투자의견 표**: 라운드별 의견 변화 상세 내역
- **최종 예측 비교**: 에이전트별 예측가 막대차트
- **라운드별 의견 변화**: 시간에 따른 예측가 변화 추이
- **반박/지지 패턴**: 에이전트 간 상호작용 분석

### 📈 시각화 기능
- 실시간 주가 차트 (7일)
- 에이전트별 예측가 비교
- 라운드별 의견 변화 추이
- 반박/지지 패턴 분석

## 🛠️ 기술 스택

### 핵심 라이브러리
- **딥러닝**: PyTorch (≥2.0.0) - LSTM 모델 학습 및 추론
- **머신러닝**: 
  - scikit-learn (≥1.0.0) - 데이터 전처리, 스케일링
  - LightGBM (≥4.0.0) - 앙상블 모델
  - SHAP (≥0.45.0) - 모델 설명가능성 분석
- **자연어처리**: Transformers (≥4.30.0) - FinBERT 감성 분석
- **LLM**: OpenAI API (≥1.0.0) - GPT-4, GPT-4o-mini

### 데이터 소스
- **주가 데이터**: Yahoo Finance API (yfinance ≥0.2.40)
- **뉴스 데이터**: EODHD API (SentimentalAgent용)
- **거시경제 데이터**: Yahoo Finance (SPY, QQQ, VIX, 금리, 환율 등)

### 시각화 및 대시보드
- **대시보드**: Streamlit (≥1.28.0)
- **차트**: Plotly (≥5.0.0), Matplotlib (≥3.5.0)

### 기타
- **언어**: Python 3.10+
- **데이터 처리**: pandas (≥1.3.0), numpy (≥1.21.0, <2.0.0)
- **설정 관리**: python-dotenv (≥0.19.0)
- **모델 저장**: joblib (≥1.0.0)

## 📁 프로젝트 구조

```
capstone/
├── agents/                          # 에이전트 모듈
│   ├── base_agent.py               # 기본 에이전트 클래스 (공통 기능)
│   ├── debate_system.py            # 토론 시스템 오케스트레이터
│   ├── macro_agent.py              # MacroAgent 구현
│   ├── technical_agent.py          # TechnicalAgent 구현
│   └── sentimental_agent.py        # SentimentalAgent 구현
│
├── config/                          # 설정 파일
│   └── agents.py                    # 에이전트 하이퍼파라미터 및 설정
│
├── core/                            # 핵심 기능 모듈
│   ├── data_set.py                 # 데이터셋 관리
│   ├── metrics.py                  # 성능 지표 계산
│   ├── backtester.py               # 백테스터 클래스
│   ├── macro_classes/              # 매크로 분석 관련 클래스
│   │   ├── macro_llm.py            # LLM 설명기 (Opinion, Rebuttal 등)
│   │   ├── macro_funcs.py          # 매크로 데이터 처리 함수
│   │   └── macro_class_dataset.py  # 매크로 데이터셋 클래스
│   ├── sentimental_classes/        # 센티멘탈 분석 관련 클래스
│   │   ├── eodhd_client.py         # EODHD API 클라이언트
│   │   ├── finbert_utils.py        # FinBERT 유틸리티
│   │   ├── finbert_scorer.py       # FinBERT 감성 점수 계산
│   │   ├── lstm_model.py           # Sentimental LSTM 모델
│   │   ├── news.py                 # 뉴스 데이터 클래스
│   │   ├── news_history_builder.py # 뉴스 히스토리 빌더
│   │   ├── sentiment_features.py  # 감성 피처 추출
│   │   └── pretrain_dataset_builder.py # 학습 데이터셋 빌더
│   └── technical_classes/           # 기술적 분석 관련 클래스
│       ├── technical.py            # 기술 지표 계산 함수
│       └── technical_data_set.py   # 기술적 데이터셋 클래스
│
├── backtest/                        # 백테스팅 전용 디렉토리
│   ├── data/                       # 백테스트 데이터
│   │   ├── raw/                    # 원시 데이터
│   │   ├── processed/             # 전처리된 데이터
│   │   └── backtests/              # 백테스트 결과
│   └── models/                     # 백테스트용 모델
│
├── scripts/                         # 유틸리티 스크립트
│   ├── gen_training_data.py        # 학습 데이터 생성
│   ├── train_meta_model.py         # 메타 모델 학습
│   ├── run_backtest_pro.py         # 프로 백테스트 실행
│   └── analyze_backtest.py         # 백테스트 결과 분석
│
├── notebooks/                       # Jupyter 노트북 (실험 및 테스트)
│
├── data/                            # 일반 데이터 저장소
│   └── processed/                  # 전처리된 데이터
│
├── models/                          # 학습된 모델 저장소
│   └── scalers/                     # 스케일러 파일
│
├── main.py                          # CLI 진입점
├── backtest.py                      # 백테스트 실행 스크립트
├── run_tuning.py                    # 하이퍼파라미터 튜닝 스크립트
├── streamlit_dashboard.py           # Streamlit 대시보드
├── prompts.py                       # LLM 프롬프트 정의
├── requirements.txt                 # Python 의존성 패키지
└── README.md                        # 프로젝트 문서
```

## 🎯 주요 특징

### ✨ MVP 완성
- 신뢰도 기반 Revise 알고리즘 구현
- 다중 전문 에이전트 완전 구현 (Technical, Sentimental, MacroSenti)
- 실시간 대시보드 완성
- Gradient 기반 Feature Importance 분석 (MacroSentiAgent)
- 롤링 백테스팅 기능 구현

### 🔬 과학적 접근
- **불확실성 기반 신뢰도**: Monte Carlo Dropout을 통한 예측 불확실성(σ) 계산
- **베이지안 의견 수정**: 불확실성 기반 가중치(β)를 사용한 합의 알고리즘
- **앙상블 학습**: LightGBM 기반 메타 모델로 최종 예측 통합
- **설명가능성(XAI)**: 
  - TechnicalAgent: Grad×Input, Occlusion, Attention 융합
  - MacroAgent: Integrated Gradients, Gradient × Input 분석
- **Fine-tuning**: 토론 과정에서 모델을 추가 학습하여 예측 보정

### 🎨 사용자 경험
- **직관적인 Streamlit 인터페이스**: 웹 기반 대시보드
- **실시간 진행 상황 표시**: 토론 진행 상황 실시간 모니터링
- **인터랙티브 차트**: Plotly 기반 동적 시각화
- **CLI 지원**: 스크립트 기반 자동화 및 배치 처리

## 📊 백테스팅 기능

### Rolling Backtest
롤링 윈도우 방식의 백테스팅을 통해 과거 데이터로 모델 성능을 평가합니다. 각 거래일마다 해당 시점 이전의 데이터만 사용하여 Look-ahead bias를 방지합니다.

#### 사용 방법

**기본 실행**:
```bash
# 단일 티커 백테스트 (기본 5거래일)
python backtest.py --ticker TSLA --predict-days 5

# 여러 티커 동시 백테스트
python backtest.py --tickers MSFT AZN CCEP --predict-days 5

# 시작 날짜 지정
python backtest.py --ticker TSLA --start 2024-01-01 --predict-days 10

# 라운드 수 지정
python backtest.py --ticker TSLA --predict-days 5 --rounds 3

# 분석 스킵 (빠른 실행)
python backtest.py --ticker TSLA --predict-days 5 --no-analyze
```

#### 주요 기능
- **독립된 디렉토리**: 백테스트 관련 모든 파일이 `backtest/` 폴더에 저장
- **Look-ahead bias 방지**: 각 거래일마다 해당 날짜 이전 데이터만 사용 (데이터 누수 방지)
- **자동 데이터 필터링**: 시뮬레이션 날짜 이전 데이터만 포함하는 임시 CSV 생성
- **자동 모델 정리**: 각 날짜 처리 후 모델 파일 자동 삭제 (Full Retraining 보장)
- **자동 분석**: 백테스트 완료 후 성능 지표 계산 및 시각화 자동 수행
- **다중 라운드 지원**: DebateSystem의 여러 라운드 토론 시뮬레이션
- **LLM 호출 스킵**: 백테스트 모드에서는 LLM 호출 없이 수치 예측만 수행 (비용 절감)
- **결과 저장**: CSV 형식으로 백테스트 결과 저장

#### 성능 지표
백테스트 결과에서 다음 지표를 확인할 수 있습니다:
- **MSE, MAE**: 예측 오차 지표
- **Direction Accuracy**: 방향 정확도 (%)
- **Strategy Return**: 전략 수익률 (%)
- **Buy & Hold Return**: 바이앤홀드 수익률 (%)
- **에이전트별 정확도**: 각 에이전트의 방향 정확도

#### 출력 파일
- **백테스트 결과**: `backtest/data/backtests/rolling_{TICKER}_{START}_{END}.csv`
- **분석 차트**: `backtest/data/backtests/analysis/`
  - `*_price.png`: 가격 예측 차트 (실제 vs 예측)
  - `*_return.png`: 누적 수익률 차트 (전략 vs 바이앤홀드)

### 파라미터 최적화 (Hyperparameter Tuning)

`run_tuning.py`를 사용하여 그리드 서치 방식으로 하이퍼파라미터를 최적화할 수 있습니다. 각 파라미터 조합에 대해 백테스트를 수행하고 최적의 설정을 찾습니다.

#### 사용 방법

**기본 사용법**:
```bash
# 기본 설정으로 튜닝 실행 (config/agents.py에서 티커 목록 확인)
python run_tuning.py
```

**주요 기능**:
- **그리드 서치**: 모든 파라미터 조합을 테스트하여 최적의 설정 탐색
- **자동 실험 관리**: 각 실험마다 모델과 데이터를 자동으로 정리하고 재생성
- **실시간 결과 저장**: 실험 진행 중간에도 결과를 CSV 파일로 저장
- **다중 티커 지원**: 여러 종목에 대해 순차적으로 튜닝 수행
- **로그 관리**: 터미널 출력과 파일 출력을 동시에 기록

**튜닝되는 파라미터**:
- **공통 파라미터**: `fine_tune_lr` (Fine-tuning 학습률)
- **TechnicalAgent**: 
  - `window_size` (시계열 윈도우 크기)
  - `rnn_units1` (첫 번째 LSTM 레이어 유닛 수)
  - `learning_rate` (학습률)
- **SentimentalAgent**: 
  - `window_size` (시계열 윈도우 크기)
  - `d_model` (LSTM hidden dimension)
  - `learning_rate` (학습률)
- **MacroAgent**: 
  - `window_size` (시계열 윈도우 크기)
  - `learning_rate` (학습률)

**출력 파일**:
- **튜닝 요약**: `backtest/tuning_results/{TICKER}/tuning_summary_{TICKER}_{TIMESTAMP}.csv`
  - 각 실험의 파라미터 조합과 성능 지표 요약
- **실험별 결과**: `backtest/tuning_results/{TICKER}/exp_{N}/rolling_{TICKER}_{START}_{END}.csv`
  - 각 실험의 상세 백테스트 결과
- **실행 로그**: `backtest/tuning_results/{TICKER}/tuning_log_{TICKER}_{TIMESTAMP}.txt`
  - 실험 진행 상황 및 오류 로그
- **전체 요약**: `backtest/tuning_results/total_best_summary_{TIMESTAMP}.csv`
  - 모든 티커에 대한 최적 파라미터 요약

**결과 분석**:
튜닝 완료 후 생성된 CSV 파일에서 다음 지표를 확인할 수 있습니다:
- `strategy_return`: 전략 수익률 (%)
- `buy_hold_return`: 바이앤홀드 수익률 (%)
- `direction_acc`: 방향 정확도 (%)
- `mse`, `mae`: 예측 오차 지표
- 각 에이전트별 정확도 (`TechnicalAgent_acc`, `SentimentalAgent_acc`, `MacroAgent_acc`)

**주의사항**:
- ⚠️ 튜닝은 시간이 오래 걸릴 수 있습니다 (각 실험마다 모델 학습 및 백테스트 수행)
- 실험 중간에 중단되더라도 이미 완료된 실험 결과는 저장되어 있습니다
- 로그 파일을 확인하여 진행 상황을 모니터링할 수 있습니다
- 최적 파라미터는 종목별로 다를 수 있으므로, 각 티커에 대해 별도로 튜닝하는 것을 권장합니다

## 📝 추가 정보

### 모델 학습 프로세스
1. **Pretrain**: 각 에이전트가 독립적으로 모델 학습
   - 데이터 수집 및 전처리
   - LSTM 모델 학습 (Early Stopping 적용)
   - 모델 저장 및 검증
2. **Predict**: 학습된 모델로 예측 수행
   - Monte Carlo Dropout으로 불확실성 추정
   - LLM을 통한 예측 근거 생성
3. **Revise**: 토론 후 예측 수정
   - Fine-tuning을 통한 모델 보정
   - 신뢰도 기반 가중 평균으로 합의 가격 계산
4. **Ensemble**: 최종 앙상블 예측
   - LightGBM 메타 모델로 통합 예측

### 데이터 흐름
```
Raw Data (Yahoo Finance, EODHD)
    ↓
Searcher (각 에이전트별 데이터 수집)
    ↓
Preprocessing (스케일링, 피처 엔지니어링)
    ↓
Model Training (Pretrain)
    ↓
Prediction (Monte Carlo Dropout)
    ↓
LLM Explanation (Opinion, Rebuttal, Revise)
    ↓
Ensemble (LightGBM)
    ↓
Final Prediction
```

## 🚀 향후 계획

- [ ] 더 많은 에이전트 추가 (Quantitative, ESG 등)
- [ ] 백테스팅 기능 고도화 (포트폴리오 최적화, 리스크 관리 등)
- [ ] 실시간 알림 기능 (예측 변동 시 알림)
- [ ] API 서비스 제공 (REST API)
- [ ] 모바일 앱 개발
- [ ] 다국어 지원 (영어, 일본어 등)

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 🤝 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다!

