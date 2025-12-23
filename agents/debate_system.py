# agents/debate_system.py
"""
DebateSystem: Multi-Agent Debate System Orchestrator

이 모듈은 여러 에이전트(TechnicalAgent, MacroAgent, SentimentalAgent) 간의
토론을 조율하고 최종 예측을 생성하는 역할을 담당합니다.

주요 기능:
- Opinion 수집: 각 에이전트로부터 초기 예측 및 근거 수집
- Rebuttal 생성: 에이전트 간 상호 반박 및 지지 메시지 생성
- Revision: 토론 내용을 바탕으로 예측 수정 (합의 알고리즘 + Fine-tuning)
- Ensemble: 최종적으로 수렴된 의견을 통합하여 Ensemble 예측 생성
"""
import os
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from scripts.train_meta_model import directional_mse_objective
import traceback
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statistics
from datetime import timedelta
import yfinance as yf

from config.agents_set import dir_info, agents_info, common_params
from config.prompts import DEBATE_PROMPTS

from agents.macro_agent import MacroAgent
from agents.technical_agent import TechnicalAgent
from agents.sentimental_agent import SentimentalAgent
from core.technical_classes.technical_data_set import load_dataset as load_dataset_tech
from core.macro_classes.macro_llm import Opinion, Rebuttal


class DebateSystem:
    """
    Multi-Agent Debate System Orchestrator
    여러 에이전트 간의 토론 프로세스(Round 0 ~ N)를 관리하고 최종 예측을 도출합니다.
    """

    def __init__(self, ticker: str, rounds: int = 3, data_dir: Optional[str] = None, model_dir: Optional[str] = None):
        """
        DebateSystem 초기화

        Args:
            ticker (str): 분석할 종목 티커 (예: "NVDA")
            rounds (int): 진행할 토론 라운드 수 (기본값: 3)
            data_dir (str): 데이터 저장 경로 (None이면 config 기본값 사용)
            model_dir (str): 모델 저장 경로 (None이면 config 기본값 사용)
        """
        if not ticker or str(ticker).strip() == "":
            raise ValueError("DebateSystem: ticker must not be None or empty")

        self.ticker = str(ticker).upper()
        
        # 경로 설정
        self.data_dir = data_dir if data_dir is not None else dir_info["data_dir"]
        self.model_dir = model_dir if model_dir is not None else dir_info["model_dir"]
        self.scaler_dir = os.path.join(self.model_dir, "scalers")

        # OpenAI Client 초기화
        load_dotenv()
        self.openai_api_key = os.getenv("CAPSTONE_OPENAI_API")
        self.client = None
        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
            except Exception as e:
                print(f"[WARN] OpenAI Client 초기화 실패: {e}")

        # Config 로드
        macro_cfg = agents_info.get("MacroAgent", {})
        macro_window = macro_cfg.get("window_size", 40)
        tech_cfg = agents_info.get("TechnicalAgent", {})
        sent_cfg = agents_info.get("SentimentalAgent", {})
        
        # 에이전트 인스턴스 생성
        self.agents = {
            "TechnicalAgent": TechnicalAgent(
                agent_id="TechnicalAgent",
                ticker=self.ticker,
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                gamma=tech_cfg.get("gamma", 0.3),
                delta_limit=tech_cfg.get("delta_limit", 0.05)
            ),

            "MacroAgent": MacroAgent(
                agent_id="MacroAgent",
                ticker=self.ticker,
                base_date=datetime.today(),
                window=macro_window,
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                gamma=macro_cfg.get("gamma", 0.5),
                delta_limit=macro_cfg.get("delta_limit", 0.1)
            ),

            "SentimentalAgent": SentimentalAgent(
                ticker=self.ticker,
                agent_id="SentimentalAgent",
                data_dir=self.data_dir,
                model_dir=self.model_dir,
                news_dir=None,  # 자동 설정
                gamma=sent_cfg.get("gamma", 0.3),
                delta_limit=sent_cfg.get("delta_limit", 0.05)
            ),
        }

        # Debate 상태 관리
        self.rounds = rounds
        self.opinions: Dict[int, Dict[str, Opinion]] = {}
        self.rebuttals: Dict[int, List[Rebuttal]] = {}

        # Ensemble 모델 (LightGBM)은 run() 시점에 로드/학습
        self.ensemble_model = None

    def _check_agent_ready(self, agent_id: str, ticker: str) -> bool:
        """
        에이전트의 모델 및 스케일러 파일이 존재하는지 확인합니다.
        """
        model_path = os.path.join(self.model_dir, f"{ticker}_{agent_id}.pt")

        if not os.path.exists(model_path):
            return False

        if agent_id == "MacroAgent":
            scaler_X_path = os.path.join(self.model_dir, "scalers", f"{ticker}_{agent_id}_xscaler.pkl")
            scaler_y_path = os.path.join(self.model_dir, "scalers", f"{ticker}_{agent_id}_yscaler.pkl")
            if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
                return False

        return True

    def get_opinion(self, round: int, ticker: str = None, rebuild: bool = False, force_pretrain: bool = False):
        """
        각 에이전트로부터 초기 의견(Opinion)을 수집합니다.
        
        Args:
            round (int): 현재 라운드 번호 (0부터 시작)
            ticker (str): 종목 코드
            rebuild (bool): 데이터셋 강제 재생성 여부
            force_pretrain (bool): 강제 재학습 여부
            
        Returns:
            Dict[str, Opinion]: 에이전트 ID를 키로 하는 의견 딕셔너리
        """
        ticker = ticker or self.ticker
        if not ticker:
            raise ValueError("ticker가 지정되지 않았습니다.")

        opinions = {}

        for agent_id, agent in self.agents.items():
            # 1. 모델 상태 확인
            is_ready = self._check_agent_ready(agent_id, ticker)
            needs_pretrain = force_pretrain or (not is_ready)

            # 2. 데이터 준비 및 학습 (필요시)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] search 실행 (데이터셋 준비)")
            X = agent.search(ticker, rebuild=rebuild)
            
            if needs_pretrain:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] pretrain 실행 (모델/스케일러 생성)")
                agent.pretrain()
            
            # 3. 예측 수행
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] predict 실행")
            n_samples = common_params.get("n_samples", 30) if agent_id == "SentimentalAgent" else None
            if n_samples:
                target = agent.predict(agent.stockdata, n_samples=n_samples)
            else:
                target = agent.predict(agent.stockdata)

            # 4. Opinion 생성 (LLM)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] review_draft 실행")
            opinion = agent.review_draft(agent.stockdata, target)

            opinions[agent_id] = opinion
            try:
                print(f"  - {agent_id}: next_close={opinion.target.next_close:.4f}")
            except Exception:
                pass

        self.opinions[round] = opinions
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 의견 수집 완료 ({len(opinions)} agents)")
        return opinions


    def get_rebuttal(self, round: int):
        """
        이전 라운드의 의견에 대해 에이전트 간 상호 반박(Rebuttal)을 생성합니다.
        
        Args:
            round (int): 현재 라운드 번호
            
        Returns:
            List[Rebuttal]: 생성된 반박 메시지 리스트
        """
        round_rebuttals = []
        prev_round = round - 1
        
        if prev_round not in self.opinions:
            raise ValueError(f"이전 라운드({prev_round})의 의견이 없습니다.")

        opinions = self.opinions[prev_round]

        for agent_id, agent in self.agents.items():
            my_opinion = opinions[agent_id]

            # 타 에이전트의 의견에 대해 반박/지지 생성
            for other_id, other_op in opinions.items():
                if other_id == agent_id:
                    continue

                print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] → [{other_id}] rebuttal 생성 중...")
                rebut = agent.review_rebut(
                    my_opinion=my_opinion,
                    other_opinion=other_op,
                    round=round,
                )
                round_rebuttals.append(rebut)

        self.rebuttals[round] = round_rebuttals
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} rebuttal 완료 ({len(round_rebuttals)}개)")
        return round_rebuttals


    def get_revise(self, round: int):
        """
        반박 내용을 반영하여 각 에이전트가 자신의 예측을 수정(Revise)합니다.
        합의 알고리즘 적용 및 Fine-tuning이 포함될 수 있습니다.
        
        Args:
            round (int): 현재 라운드 번호
            
        Returns:
            Dict[str, Opinion]: 수정된 의견 딕셔너리
        """
        if (round - 1) not in self.opinions:
            raise ValueError(f"이전 라운드({round-1})의 의견이 없습니다.")

        round_revises = {}

        for agent_id, agent in self.agents.items():
            my_opinion = self.opinions[round - 1][agent_id]
            other_opinions = [
                self.opinions[round - 1][other_id]
                for other_id in self.agents.keys()
                if other_id != agent_id
            ]
            rebuttals = [
                r for r in self.rebuttals.get(round, [])
                if getattr(r, "to_agent_id", None) == agent_id
            ]
            stock_data = getattr(agent, "stockdata", None)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{agent_id}] revise 실행 중...")
            
            revised_opinion = agent.review_revise(
                my_opinion=my_opinion,
                others=other_opinions,
                rebuttals=rebuttals,
                stock_data=stock_data,
            )

            round_revises[agent_id] = revised_opinion

        self.opinions[round] = round_revises
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} revise 완료 ({len(round_revises)} agents)")

        return round_revises

    def run(self, force_pretrain: bool = False):
        """
        전체 디베이트 프로세스를 실행합니다.
        
        Sequence:
        1. Round 0: 초기 예측 (Opinion)
        2. Round 1 ~ N: 토론 (Rebuttal) -> 수정 (Revise) 반복
        3. Final: Ensemble 예측 생성
        
        Args:
            force_pretrain (bool): 초기화 시 강제 재학습 여부
        """
        # Round 0: 초기 Opinion 수집
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Round 0: 초기 Opinion 수집 시작 (force_pretrain={force_pretrain})")
        print(f"{'='*80}")
        self.get_opinion(0, self.ticker, rebuild=False, force_pretrain=force_pretrain)

        # Round 1~N: Rebuttal → Revise 반복
        for round in range(1, self.rounds + 1):
            print(f"\n{'='*80}")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 시작")
            print(f"{'='*80}")

            self.get_rebuttal(round)
            self.get_revise(round)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Round {round} 토론 완료")

        # 최종 Ensemble 예측
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 최종 Ensemble 예측")
        print(f"{'='*80}")
        ensemble_result = self.get_ensemble()
        print(ensemble_result)

        return ensemble_result


    def summarize_debate(self, ensemble_result: Dict) -> str:
        """
        전체 토론 과정을 요약하고 최종 결론을 도출합니다. (LLM 활용)
        """
        # 백테스팅 모드 확인 (요약 스킵)
        is_backtest = False
        for agent in self.agents.values():
            if hasattr(agent, 'test_mode') and agent.test_mode:
                is_backtest = True
                break
        
        if is_backtest:
            return "[백테스팅 모드] Debate 요약 스킵됨"

        if not self.client:
            return "OpenAI API 키가 설정되지 않아 요약을 생성할 수 없습니다."

        # 1. Transcript 생성
        transcript_lines = []
        
        # Round 0
        transcript_lines.append("\n## [Round 0: 초기 의견]")
        if 0 in self.opinions:
            for agent_id, op in self.opinions[0].items():
                transcript_lines.append(f"- {agent_id}: 예측가 {op.target.next_close:.2f}, 근거: {op.reason}")
                
        # Round 1 ~ N
        for r in range(1, self.rounds + 1):
            transcript_lines.append(f"\n## [Round {r}: 토론 및 수정]")
            
            if r in self.rebuttals:
                transcript_lines.append("### 반박(Rebuttals):")
                for reb in self.rebuttals[r]:
                    transcript_lines.append(
                        f"  * {reb.from_agent_id} -> {reb.to_agent_id} ({reb.stance}): {reb.message}"
                    )
            
            if r in self.opinions:
                transcript_lines.append("### 수정된 의견(Revised Opinions):")
                for agent_id, op in self.opinions[r].items():
                    transcript_lines.append(f"  * {agent_id}: 예측가 {op.target.next_close:.2f}, 근거: {op.reason}")

        transcript = "\n".join(transcript_lines)
        
        # 2. Prompt 구성
        prompts = DEBATE_PROMPTS["summary"]
        system_msg = prompts["system"]
        user_msg = prompts["user_template"].format(
            transcript=transcript,
            ticker=ensemble_result.get("ticker", "Unknown"),
            last_price=ensemble_result.get("last_price", "N/A"),
            ensemble_price=f"{ensemble_result.get('ensemble_next_close', 0.0):.2f}",
            return_pct=f"{(ensemble_result.get('ensemble_next_close', 0.0) / ensemble_result.get('last_price', 1.0) - 1) * 100:.2f}" if ensemble_result.get('last_price') else "N/A"
        )

        # 3. LLM 호출
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] 요약 생성 중 오류 발생: {e}")
            return f"요약 생성 실패: {e}"




    def get_ensemble(self) -> Dict:
        """
        토론 결과를 바탕으로 ensemble 정보 생성 (LightGBM Model 적용)

        Returns:
            Dict: Ensemble 예측 정보
                - ticker: 종목 코드
                - agents: 에이전트별 예측가 딕셔너리
                - mean_next_close: (모델 미사용시) 평균 예측가
                - median_next_close: (모델 미사용시) 중앙값 예측가
                - ensemble_next_close: (모델 사용시) 최종 모델 예측가
                - currency: 통화 코드
                - last_price: 현재가
        """
        # -------------------------------------------------------
        # 앙상블 모델 준비 (없으면 자동 학습)
        # -------------------------------------------------------
        if not hasattr(self, 'ensemble_model') or self.ensemble_model is None:
            model_filename = f"{self.ticker}_ensemble.pt"
            model_path = os.path.join(self.model_dir, model_filename)
            data_path = os.path.join(self.data_dir, f"{self.ticker}_ensemble_train.csv")

            # 모델이 없으면 학습 시작
            if not os.path.exists(model_path):
                print(f"\n{'='*60}")
                print(f"[INFO] {self.ticker} 전용 앙상블 모델이 없습니다. 자동 학습을 시작합니다.")
                print(f"{'='*60}")

                try:
                    # 1. 학습 데이터 생성 (무조건 재생성)
                    print(f"[Step 1/3] 학습 데이터 생성 중... ({self.ticker})")

                    # 기존 파일이 있으면 삭제
                    if os.path.exists(data_path):
                        os.remove(data_path)
                        print(f"  기존 파일 삭제: {data_path}")

                    # 1-1. 학습된 모델/스케일러 로드
                    print("  1-1. 학습된 모델/스케일러 로드 중...")

                    # 기존 인스턴스 재사용 및 모델 로드 (통합된 load_model 사용)
                    for agent_id, agent in self.agents.items():
                        if not hasattr(agent, "model_loaded") or not agent.model_loaded:
                            agent.load_model()

                    print("  모델/스케일러 로드 완료.")

                    # 1-2. 입력 데이터 로드 (data/processed에서)
                    print("  1-2. 입력 데이터 로드 중 (data/processed)...")

                    # TechnicalAgent Data
                    tech_dataset_path = os.path.join(self.data_dir, f"{self.ticker}_TechnicalAgent_dataset.csv")
                    if not os.path.exists(tech_dataset_path):
                        raise FileNotFoundError(f"TechnicalAgent 데이터셋이 없습니다: {tech_dataset_path}")

                    tech_X_all, tech_y_all, tech_cols, tech_dates = load_dataset_tech(
                        self.ticker, agent_id="TechnicalAgent", save_dir=self.data_dir
                    )

                    # tech_dates 구조 확인 및 평탄화
                    tech_last_dates = []
                    if tech_dates is not None and len(tech_dates) > 0:
                        if isinstance(tech_dates[0], (list, tuple, np.ndarray)):
                            tech_last_dates = [d[-1] for d in tech_dates]
                        elif isinstance(tech_dates[0], str):
                            tech_last_dates = tech_dates

                    # MacroAgent Data (dataset.csv 사용)
                    macro_dataset_path = os.path.join(self.data_dir, f"{self.ticker}_MacroAgent_dataset.csv")
                    if not os.path.exists(macro_dataset_path):
                        raise FileNotFoundError(f"MacroAgent 데이터셋이 없습니다: {macro_dataset_path}")

                    macro_df = pd.read_csv(macro_dataset_path)
                    if 'date' in macro_df.columns:
                        # 날짜 형식 통일 (normalize로 시간 제거)
                        macro_df['date'] = pd.to_datetime(macro_df['date'], errors='coerce').dt.normalize()
                        macro_df = macro_df.sort_values(['sample_id', 'time_step'])
                        # 결측치 제거
                        macro_df = macro_df[macro_df['date'].notna()].copy()

                    # SentimentalAgent Data (dataset.csv 사용)
                    senti_dataset_path = os.path.join(self.data_dir, f"{self.ticker}_SentimentalAgent_dataset.csv")
                    if not os.path.exists(senti_dataset_path):
                        raise FileNotFoundError(f"SentimentalAgent 데이터셋이 없습니다: {senti_dataset_path}")

                    senti_df = pd.read_csv(senti_dataset_path)
                    if 'date' in senti_df.columns:
                        senti_df['date'] = pd.to_datetime(senti_df['date'], errors='coerce')
                        senti_df = senti_df.sort_values(['sample_id', 'time_step'])

                    # 가격 데이터 다운로드 (실제 종가 확인용)
                    print("  1-3. 가격 데이터 다운로드 중...")
                    period_str = common_params.get("period", "2y")
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

                    end_date = datetime.today()
                    start_date = end_date - timedelta(days=days + 60)
                    df_price = yf.download(self.ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
                    if isinstance(df_price.columns, pd.MultiIndex):
                        df_price.columns = [c[0] for c in df_price.columns]
                    df_price = df_price.reset_index()
                    df_price['Date'] = pd.to_datetime(df_price['Date']).dt.normalize()
                    df_price = df_price.sort_values('Date')

                    # 1-4. 일별 예측 수행
                    print(f"  1-4. 일별 예측 수행 중...")
                    results = []

                    # 에이전트별 window_size 저장
                    agent_windows = {agent_id: agent.window_size for agent_id, agent in self.agents.items()}
                    w_tech = agent_windows.get("TechnicalAgent", 20)
                    w_macro = agent_windows.get("MacroAgent", 40)
                    w_senti = agent_windows.get("SentimentalAgent", 30)

                    # TechnicalAgent: 날짜별 매칭
                    for t_idx, tech_date_list in enumerate(tech_dates):
                        if not tech_date_list:
                            continue
                        curr_date_str = tech_date_list[-1] if isinstance(tech_date_list, list) else str(tech_date_list)
                        curr_date = pd.to_datetime(curr_date_str).normalize()

                        # 가격 데이터에서 해당 날짜 찾기
                        price_row = df_price[df_price['Date'] == curr_date]
                        if price_row.empty:
                            continue
                        idx_price = price_row.index[0]
                        curr_close = float(price_row['Close'].iloc[0])

                        if idx_price + 1 >= len(df_price):
                            continue
                        next_close_actual = float(df_price.iloc[idx_price + 1]['Close'])

                        # Technical Prediction
                        pred_tech = np.nan; conf_tech = 0; unc_tech = 0; ret_tech = np.nan
                        try:
                            tech_agent = self.agents["TechnicalAgent"]
                            X_batch = tech_X_all[t_idx]
                            X_in = np.expand_dims(X_batch, axis=0)
                            target_tech = tech_agent.predict(X_in, current_price=curr_close)
                            pred_tech = target_tech.next_close
                            conf_tech = target_tech.confidence
                            unc_tech = target_tech.uncertainty
                            ret_tech = getattr(target_tech, "predicted_return", (pred_tech - curr_close)/curr_close if not np.isnan(pred_tech) else np.nan)
                        except Exception as e:
                            pass

                        # Macro Prediction (해당 날짜의 sample 찾기) - 다른 에이전트와 동일하게 원본 데이터 전달
                        pred_macro = np.nan; conf_macro = 0; unc_macro = 0; ret_macro = np.nan
                        try:
                            # 날짜 형식 통일 (normalize로 시간 제거)
                            curr_date_normalized = pd.to_datetime(curr_date).normalize() if not isinstance(curr_date, pd.Timestamp) else curr_date.normalize()

                            # macro_df의 date 컬럼도 normalize
                            if 'date' in macro_df.columns:
                                macro_df_date_norm = pd.to_datetime(macro_df['date'], errors='coerce').dt.normalize()
                                macro_samples = macro_df[macro_df_date_norm == curr_date_normalized]['sample_id'].unique()
                            else:
                                macro_samples = []

                            if len(macro_samples) > 0:
                                macro_sample_id = macro_samples[0]
                                macro_sample = macro_df[macro_df['sample_id'] == macro_sample_id].sort_values('time_step')
                                if len(macro_sample) >= w_macro:
                                    # 피처 컬럼 추출 (sample_id, time_step, target, date 제외)
                                    feat_cols = [c for c in macro_sample.columns if c not in ['sample_id', 'time_step', 'target', 'date']]
                                    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(macro_sample[c])]

                                    if len(feat_cols) == 0:
                                        if t_idx < 5:
                                            print(f"  [DEBUG] Macro: 피처 컬럼이 없습니다 (날짜: {curr_date_normalized})")
                                    else:
                                        X_values = macro_sample[feat_cols].values[-w_macro:]

                                        # ★ batch 차원 추가 (중요)
                                        X_values = np.expand_dims(X_values, axis=0)  # (1, w_macro, feature_dim)
                                        macro_agent = self.agents["MacroAgent"]
                                        target_macro = macro_agent.predict(X_values, current_price=curr_close)

                                        pred_macro = target_macro.next_close
                                        conf_macro = target_macro.confidence
                                        unc_macro = target_macro.uncertainty
                                        ret_macro = getattr(target_macro, "predicted_return", (pred_macro - curr_close)/curr_close if not np.isnan(pred_macro) else np.nan)
                            else:
                                # 디버깅: 날짜 매칭 실패 시 정보 출력
                                if t_idx < 5:
                                    available_dates = pd.to_datetime(macro_df['date'], errors='coerce').dt.normalize().unique() if 'date' in macro_df.columns else []
                                    print(f"  [DEBUG] Macro: 날짜 매칭 실패 (찾는 날짜: {curr_date_normalized}, 사용 가능한 날짜 수: {len(available_dates)})")
                        except Exception as e:
                            # 디버깅을 위해 에러 메시지 출력 (첫 몇 개만)
                            if t_idx < 5:
                                print(f"  [DEBUG] Macro 예측 실패 (날짜: {curr_date}): {e}")

                                traceback.print_exc()
                            pass

                        # Sentimental Prediction (해당 날짜의 sample 찾기)
                        pred_senti = np.nan; conf_senti = 0; unc_senti = 0; ret_senti = np.nan
                        try:
                            senti_samples = senti_df[senti_df['date'] == curr_date]['sample_id'].unique()
                            if len(senti_samples) > 0:
                                senti_sample_id = senti_samples[0]
                                senti_sample = senti_df[senti_df['sample_id'] == senti_sample_id].sort_values('time_step')
                                if len(senti_sample) >= w_senti:
                                    # 피처 컬럼 추출
                                    feat_cols = [c for c in senti_sample.columns if c not in ['sample_id', 'time_step', 'target', 'date']]
                                    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(senti_sample[c])]

                                    # 윈도우 데이터 추출
                                    X_values = senti_sample[feat_cols].values[-w_senti:]
                                    X_in = np.expand_dims(X_values, axis=0)
                                    senti_agent = self.agents["SentimentalAgent"]
                                    target_senti = senti_agent.predict(X_in, current_price=curr_close)
                                    pred_senti = target_senti.next_close
                                    conf_senti = target_senti.confidence
                                    unc_senti = target_senti.uncertainty
                                    ret_senti = getattr(target_senti, "predicted_return", (pred_senti - curr_close)/curr_close if not np.isnan(pred_senti) else np.nan)
                        except Exception as e:
                            pass

                        # 결과 저장
                        row = {
                            "Date": curr_date,
                            "Last_Close": curr_close,
                            "Next_Close": next_close_actual,
                            "Tech_Pred": pred_tech,
                            "Tech_Conf": conf_tech,
                            "Tech_Unc": unc_tech,
                            "Tech_Ret": ret_tech,
                            "Macro_Pred": pred_macro,
                            "Macro_Conf": conf_macro,
                            "Macro_Unc": unc_macro,
                            "Macro_Ret": ret_macro,
                            "Senti_Pred": pred_senti,
                            "Senti_Conf": conf_senti,
                            "Senti_Unc": unc_senti,
                            "Senti_Ret": ret_senti
                        }
                        results.append(row)

                    # 1-5. 예측 데이터 통합 및 결측치 제거
                    print("  1-5. 예측 데이터 통합 및 결측치 제거 중...")
                    df_out = pd.DataFrame(results)

                    # 최소 2개 이상의 에이전트 예측이 있어야 유효한 데이터로 간주
                    pred_cols = ['Tech_Pred', 'Macro_Pred', 'Senti_Pred']
                    df_out['valid_pred_count'] = df_out[pred_cols].notna().sum(axis=1)
                    df_final = df_out[df_out['valid_pred_count'] >= 2].drop(columns=['valid_pred_count'])

                    # 필수 컬럼(Last_Close, Next_Close)의 결측치 제거
                    df_final = df_final.dropna(subset=['Last_Close', 'Next_Close'])

                    output_dir = os.path.dirname(data_path)
                    os.makedirs(output_dir, exist_ok=True)
                    df_final.to_csv(data_path, index=False)
                    print(f"  학습 데이터 저장 완료: {len(df_final)}행 (결측치 제거 전: {len(df_out)}행)")

                    if len(df_final) == 0:
                        print(f"  [WARN] 유효한 데이터가 없습니다. 예측 실패 원인을 확인하세요.")
                        # 디버깅용 원본 데이터 저장
                        debug_path = data_path.replace('.csv', '_debug.csv')
                        df_out.to_csv(debug_path, index=False)
                        print(f"  디버깅용 원본 데이터 저장: {debug_path}")

                    # 2. 모델 학습
                    print(f"[Step 2/3] LightGBM 모델 학습 중...")
                    if not os.path.exists(data_path):
                        raise FileNotFoundError(f"학습 데이터 파일이 없습니다: {data_path}")

                    df = pd.read_csv(data_path)

                    if len(df) == 0:
                        raise ValueError(f"학습 데이터가 비어있습니다: {data_path}")

                    # Feature Engineering
                    # 데이터셋에 이미 *_Ret 컬럼이 있으면 사용, 없으면 계산
                    if 'Tech_Ret' in df.columns:
                        df['Tech_Ret'] = df['Tech_Ret'].fillna((df['Tech_Pred'] - df['Last_Close']) / df['Last_Close'])
                    else:
                        df['Tech_Ret'] = ((df['Tech_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)

                    if 'Macro_Ret' in df.columns:
                        df['Macro_Ret'] = df['Macro_Ret'].fillna((df['Macro_Pred'] - df['Last_Close']) / df['Last_Close'])
                    else:
                        df['Macro_Ret'] = ((df['Macro_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)

                    if 'Senti_Ret' in df.columns:
                        df['Senti_Ret'] = df['Senti_Ret'].fillna((df['Senti_Pred'] - df['Last_Close']) / df['Last_Close'])
                    else:
                        df['Senti_Ret'] = ((df['Senti_Pred'] - df['Last_Close']) / df['Last_Close']).fillna(0.0)

                    df['Tech_Conf'] = df['Tech_Conf'].fillna(0.0)
                    df['Tech_Unc'] = df['Tech_Unc'].fillna(0.0)
                    df['Macro_Conf'] = df['Macro_Conf'].fillna(0.0)
                    df['Macro_Unc'] = df['Macro_Unc'].fillna(0.0)
                    df['Senti_Conf'] = df['Senti_Conf'].fillna(0.0)
                    df['Senti_Unc'] = df['Senti_Unc'].fillna(0.0)

                    df['Target_Ret'] = (df['Next_Close'] - df['Last_Close']) / df['Last_Close']

                    feature_cols = [
                        'Tech_Ret', 'Tech_Conf', 'Tech_Unc',
                        'Macro_Ret', 'Macro_Conf', 'Macro_Unc',
                        'Senti_Ret', 'Senti_Conf', 'Senti_Unc'
                    ]

                    df_clean = df.dropna(subset=['Target_Ret'])

                    if len(df_clean) == 0:
                        raise ValueError("Target_Ret가 모두 결측치입니다. 학습 데이터를 확인하세요.")

                    X = df_clean[feature_cols]
                    y = df_clean['Target_Ret']

                    # LightGBM 학습 (하이퍼파라미터는 config에서 로드)
                    model = lgb.LGBMRegressor(
                        n_estimators=common_params.get("ensemble_n_estimators", 100),
                        learning_rate=common_params.get("ensemble_learning_rate", 0.05),
                        max_depth=common_params.get("ensemble_max_depth", 3),
                        random_state=common_params.get("ensemble_random_state", 42),
                        n_jobs=common_params.get("ensemble_n_jobs", -1),
                        verbosity=common_params.get("ensemble_verbosity", -1),
                        objective=directional_mse_objective  # Custom Objective 적용
                    )

                    model.fit(
                        X, y,
                        eval_metric='mse',
                        callbacks=[lgb.log_evaluation(period=0)]  # 로그 출력 억제
                    )

                    # 모델 저장
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(model, model_path)
                    print(f"[Step 3/3] {self.ticker} 앙상블 모델 학습 완료!")

                except FileNotFoundError as e:
                    print(f"[ERROR] 앙상블 모델 학습 데이터 파일을 찾을 수 없습니다: {e}")
                    print("[INFO] 기본 평균 방식을 사용합니다.")
                    self.ensemble_model = None
                except ValueError as e:
                    print(f"[ERROR] 앙상블 모델 학습 데이터가 유효하지 않습니다: {e}")
                    print("[INFO] 기본 평균 방식을 사용합니다.")
                    self.ensemble_model = None
                except Exception as e:
                    print(f"[ERROR] 앙상블 모델 자동 학습 중 예상치 못한 오류 발생: {e}")
                    traceback.print_exc()
                    print("[INFO] 기본 평균 방식을 사용합니다.")
                    self.ensemble_model = None

            # 모델 로드 시도
            if os.path.exists(model_path):
                try:
                    self.ensemble_model = joblib.load(model_path)
                    print(f"[INFO] Ensemble Model 로드 완료: {model_path}")
                except Exception as e:
                    print(f"[WARN] Ensemble Model 로드 실패: {e}")
                    self.ensemble_model = None
            else:
                print("[WARN] 모델 파일이 생성되지 않았습니다.")
                self.ensemble_model = None

        # 최종 라운드의 의견 가져오기
        final_round = max(self.opinions.keys()) if self.opinions else 0
        final_opinions = self.opinions.get(final_round, {})

        if not final_opinions:
            print("[WARN] 최종 의견이 없습니다.")
            return {
                "ticker": self.ticker,
                "agents": {},
                "mean_next_close": None,
                "median_next_close": None,
                "ensemble_next_close": None,
                "currency": "USD",
                "last_price": None,
            }

        # 동적으로 에이전트별 의견 가져오기
        tech_op = final_opinions.get("TechnicalAgent")
        macro_op = final_opinions.get("MacroAgent")
        senti_op = final_opinions.get("SentimentalAgent")

        # 현재가(Last Price) 가져오기 (우선 에이전트가 가진 정보 활용)
        last_price = None

        # 각 에이전트에서 last_price 찾기 시도
        for agent_id, agent in self.agents.items():
            sd = getattr(agent, "stockdata", None)
            if sd and getattr(sd, "last_price", None):
                last_price = float(sd.last_price)
                break

        # 없다면 yfinance 호출 (fallback)
        if last_price is None:
            try:
                stock = yf.Ticker(self.ticker)
                info = stock.info
                last_price = info.get('currentPrice', info.get('regularMarketPrice', None))
            except:
                pass

        if last_price is None:
            print("[WARN] 현재가(Last Price)를 찾을 수 없어 Ensemble Model을 실행할 수 없습니다.")

        # 1. 모델 기반 예측 시도
        ensemble_price = None

        if self.ensemble_model and last_price:
            try:
                # 입력 벡터 구성
                def get_feats(op):
                    if not op or not op.target:
                        return np.nan, 0.0, 0.0
                    # predicted_return이 있으면 직접 사용, 없으면 가격에서 역변환
                    pred = float(op.target.next_close)
                    ret = getattr(op.target, "predicted_return", None)
                    if ret is None:
                        ret = (pred - last_price) / last_price
                    else:
                        ret = float(ret)
                    conf = float(op.target.confidence or 0.0)
                    unc = float(op.target.uncertainty or 0.0)
                    return ret, conf, unc

                t_ret, t_conf, t_unc = get_feats(tech_op)
                m_ret, m_conf, m_unc = get_feats(macro_op)
                s_ret, s_conf, s_unc = get_feats(senti_op)

                # 입력 데이터프레임 (모델 학습시 feature name과 일치해야 함)
                input_df = pd.DataFrame([{
                    'Tech_Ret': t_ret, 'Tech_Conf': t_conf, 'Tech_Unc': t_unc,
                    'Macro_Ret': m_ret, 'Macro_Conf': m_conf, 'Macro_Unc': m_unc,
                    'Senti_Ret': s_ret, 'Senti_Conf': s_conf, 'Senti_Unc': s_unc
                }])

                # 예측 (Target_Ret)
                pred_ret = self.ensemble_model.predict(input_df)[0]

                # 가격 변환
                ensemble_price = last_price * (1 + pred_ret)
                print(f"[INFO] Ensemble Model Predict: {ensemble_price:.2f} (Return: {pred_ret*100:.2f}%)")

            except Exception as e:
                print(f"[WARN] Ensemble Model 예측 중 오류 발생: {e}")
                ensemble_price = None

        # 2. 기존 통계 기반 집계 (Backup)
        final_points = [
            float(op.target.next_close)
            for op in final_opinions.values()
            if op and op.target
        ]

        mean_val = statistics.fmean(final_points) if final_points else None
        median_val = statistics.median(final_points) if final_points else None

        # 모델 예측이 실패했거나 없으면 평균값 사용
        if ensemble_price is None:
            ensemble_price = mean_val

        # 결과 구성
        agents_data = {}
        for agent_id, opinion in final_opinions.items():
            if opinion and opinion.target:
                agents_data[f"{agent_id}_next_close"] = float(opinion.target.next_close)

        # 기본 결과 딕셔너리
        result = {
            "ticker": self.ticker,
            "agents": agents_data,
            "mean_next_close": mean_val,
            "median_next_close": median_val,
            "ensemble_next_close": ensemble_price,
            "currency": "USD", # 통화는 일단 USD 고정 (개선 가능)
            "last_price": last_price,
        }

        # 3. Debate Summary 생성 (LLM)
        print(f"\n{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Debate Summary 생성 중...")
        print(f"{'='*80}")
        summary_text = self.summarize_debate(result)
        result["debate_summary"] = summary_text
        # print(summary_text) # DebateSystem에서의 직접 출력은 끔

        return result
