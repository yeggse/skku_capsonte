#!/usr/bin/env python3
"""
Streamlit Dashboard for Multi-Agent Debate System

이 대시보드는 DebateSystem의 토론 결과를 시각화하고
사용자가 인터랙티브하게 파라미터를 조정할 수 있게 해줍니다.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import traceback

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.debate_system import DebateSystem
from config.agents_set import dir_info


# 페이지 설정
st.set_page_config(
    page_title="AI Stock Debate System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "debate_system" not in st.session_state:
    st.session_state.debate_system = None
if "ensemble_result" not in st.session_state:
    st.session_state.ensemble_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None


def load_stock_data(ticker: str) -> Dict:
    """
    yfinance를 사용하여 기본 주식 데이터 로드
    
    Args:
        ticker: 종목 코드
        
    Returns:
        주식 정보 딕셔너리
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 주가 데이터 (최근 90일)
        hist = stock.history(period="90d")
        
        return {
            "info": info,
            "history": hist,
            "success": True
        }
    except Exception as e:
        return {
            "info": {},
            "history": pd.DataFrame(),
            "success": False,
            "error": str(e)
        }


def render_stock_overview_tab(ticker: str):
    """탭 1: 기본 주식 데이터 렌더링"""
    st.header("📈 기본 주식 데이터")
    
    # 주식 데이터 로드
    with st.spinner(f"{ticker} 주식 데이터를 불러오는 중..."):
        stock_data = load_stock_data(ticker)
    
    if not stock_data["success"]:
        st.error(f"주식 데이터를 불러올 수 없습니다: {stock_data.get('error', 'Unknown error')}")
        return
    
    info = stock_data["info"]
    hist = stock_data["history"]
    
    if hist.empty:
        st.warning("주가 데이터를 불러올 수 없습니다.")
        return
    
    # 현재가 정보
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
    prev_close = info.get("previousClose") or hist["Close"].iloc[-2] if len(hist) > 1 else current_price
    market_cap = info.get("marketCap", 0)
    volume = info.get("volume", 0) or hist["Volume"].iloc[-1] if "Volume" in hist.columns else 0
    
    with col1:
        st.metric("현재가", f"${current_price:,.2f}" if current_price else "N/A")
    with col2:
        change = current_price - prev_close if current_price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0
        st.metric("전일 대비", f"${change:,.2f}", f"{change_pct:.2f}%")
    with col3:
        if market_cap:
            market_cap_b = market_cap / 1e9
            st.metric("시가총액", f"${market_cap_b:.2f}B")
        else:
            st.metric("시가총액", "N/A")
    with col4:
        if volume:
            volume_m = volume / 1e6
            st.metric("거래량", f"{volume_m:.2f}M")
        else:
            st.metric("거래량", "N/A")
    
    st.divider()
    
    # 주가 차트
    st.subheader("주가 차트")
    
    # 기간 선택
    period_option = st.radio("기간 선택", ["30일", "60일", "90일"], horizontal=True)
    days = int(period_option.replace("일", ""))
    
    chart_data = hist.tail(days) if len(hist) >= days else hist
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Close"],
        mode="lines",
        name="종가",
        line=dict(color="#1f77b4", width=2)
    ))
    
    fig.update_layout(
        title=f"{ticker} 주가 추이 ({period_option})",
        xaxis_title="날짜",
        yaxis_title="가격 (USD)",
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 기본 통계
    st.subheader("기본 통계")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**52주 최고/최저가**")
        week_52_high = info.get("fiftyTwoWeekHigh", "N/A")
        week_52_low = info.get("fiftyTwoWeekLow", "N/A")
        if isinstance(week_52_high, (int, float)):
            st.write(f"최고: ${week_52_high:,.2f}")
        else:
            st.write(f"최고: {week_52_high}")
        if isinstance(week_52_low, (int, float)):
            st.write(f"최저: ${week_52_low:,.2f}")
        else:
            st.write(f"최저: {week_52_low}")
    
    with col2:
        st.write("**재무 지표**")
        pe_ratio = info.get("trailingPE", "N/A")
        beta = info.get("beta", "N/A")
        st.write(f"P/E 비율: {pe_ratio}")
        st.write(f"베타: {beta}")
    
    with col3:
        st.write("**거래 정보**")
        avg_volume = info.get("averageVolume", "N/A")
        if isinstance(avg_volume, (int, float)):
            avg_volume_m = avg_volume / 1e6
            st.write(f"평균 거래량: {avg_volume_m:.2f}M")
        else:
            st.write(f"평균 거래량: {avg_volume}")


def render_final_conclusion_tab(debate_system: DebateSystem, ensemble_result: Dict):
    """탭 2: 최종 결론 및 의견 렌더링"""
    st.header("🎯 최종 결론 및 의견")
    
    if not ensemble_result:
        st.warning("토론 결과가 없습니다. 먼저 토론을 실행해주세요.")
        return
    
    # 최종 Ensemble 예측
    st.subheader("최종 Ensemble 예측")
    
    ensemble_price = ensemble_result.get("ensemble_next_close")
    last_price = ensemble_result.get("last_price")
    
    if ensemble_price and last_price:
        return_pct = (ensemble_price / last_price - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("현재가", f"${last_price:,.2f}")
        with col2:
            st.metric("예측가", f"${ensemble_price:,.2f}")
        with col3:
            color = "normal" if return_pct == 0 else ("normal" if return_pct > 0 else "inverse")
            st.metric("예상 수익률", f"{return_pct:.2f}%", delta=f"{return_pct:.2f}%")
    else:
        st.warning("예측 데이터가 불완전합니다.")
    
    st.divider()
    
    # 각 에이전트별 최종 의견
    st.subheader("에이전트별 최종 의견")
    
    final_round = max(debate_system.opinions.keys()) if debate_system.opinions else None
    
    if final_round is None:
        st.warning("에이전트 의견 데이터가 없습니다.")
        return
    
    final_opinions = debate_system.opinions.get(final_round, {})
    
    if not final_opinions:
        st.warning("최종 라운드의 의견이 없습니다.")
        return
    
    # 에이전트별 의견 데이터 수집
    opinions_data = []
    for agent_id, opinion in final_opinions.items():
        if opinion and opinion.target:
            opinions_data.append({
                "에이전트": agent_id,
                "예측가": opinion.target.next_close,
                "신뢰도": opinion.target.confidence,
                "불확실성": opinion.target.uncertainty,
                "근거": opinion.reason
            })
    
    if opinions_data:
        # 에이전트별 탭 생성
        agent_tabs = st.tabs([row["에이전트"] for row in opinions_data])
        
        for idx, (tab, row) in enumerate(zip(agent_tabs, opinions_data)):
            with tab:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("예측가", f"${row['예측가']:,.2f}")
                with col2:
                    st.metric("신뢰도", f"{row['신뢰도']:.4f}" if row['신뢰도'] else "N/A")
                with col3:
                    st.metric("불확실성", f"{row['불확실성']:.4f}" if row['불확실성'] else "N/A")
                
                st.divider()
                st.subheader("근거")
                # 근거가 JSON 형식인 경우 파싱하여 표시
                reason_text = row['근거']
                if reason_text:
                    # JSON 형식인지 확인
                    if reason_text.strip().startswith('{') and reason_text.strip().endswith('}'):
                        try:
                            import json
                            reason_dict = json.loads(reason_text)
                            if 'reason' in reason_dict:
                                st.markdown(reason_dict['reason'])
                            else:
                                st.markdown(reason_text)
                        except:
                            st.markdown(reason_text)
                    else:
                        st.markdown(reason_text)
                else:
                    st.info("근거가 없습니다.")
        
        # 예측가 비교 차트
        st.subheader("예측가 비교")
        
        agent_names = [row["에이전트"] for row in opinions_data]
        prices = [row["예측가"] for row in opinions_data]
        
        if ensemble_price:
            agent_names.append("Ensemble")
            prices.append(ensemble_price)
        
        # y축 범위 계산 (최소값 -10%, 최대값 +10%)
        min_price = min(prices)
        max_price = max(prices)
        y_min = min_price * 0.95
        y_max = max_price * 1.05
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=agent_names,
            y=prices,
            text=[f"${p:,.2f}" for p in prices],
            textposition="auto",
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(agent_names)]
        ))
        
        fig.update_layout(
            title="에이전트별 예측가 비교",
            xaxis_title="에이전트",
            yaxis_title="예측가 (USD)",
            yaxis_range=[y_min, y_max],
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Debate Summary
    ticker = debate_system.ticker if debate_system else ensemble_result.get("ticker", "TSLA")
    st.subheader(f"{ticker} 투자 토론 요약 및 결론 리포트")
    debate_summary = ensemble_result.get("debate_summary", "")
    
    if debate_summary:
        # 섹션 헤더 파싱 (예: [토론 요약], [주요 쟁점], [최종 결론 및 제언] 등)
        import re
        sections = {}
        current_section = None
        current_content = []
        
        lines = debate_summary.split('\n')
        for line in lines:
            # 섹션 헤더 패턴 찾기: [섹션명] 형식
            section_match = re.match(r'^##?\s*\[([^\]]+)\]', line)
            if section_match:
                # 이전 섹션 저장
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                # 새 섹션 시작
                current_section = section_match.group(1)
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # 섹션 헤더가 없는 경우 기본 섹션으로
                    if not current_section:
                        current_section = "전체 요약"
                        current_content = [line]
        
        # 마지막 섹션 저장
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # 섹션을 순차적으로 표시 (탭 없이)
        if sections:
            for section_name, section_content in sections.items():
                # 섹션 제목 표시 (큰 제목 제거)
                if section_name != "전체 요약":
                    st.markdown(f"### [{section_name}]")
                # 섹션 내용 표시 (큰 제목이나 불필요한 헤더 제거)
                content_lines = section_content.split('\n')
                filtered_lines = []
                for line in content_lines:
                    # "투자결론리포트" 같은 큰 제목 제거
                    if not re.match(r'^#+\s*(투자|결론|리포트)', line, re.IGNORECASE):
                        filtered_lines.append(line)
                st.markdown('\n'.join(filtered_lines))
                if section_name != list(sections.keys())[-1]:  # 마지막 섹션이 아니면 구분선
                    st.divider()
        else:
            # 섹션이 없는 경우 전체 텍스트 표시 (큰 제목 제거)
            content_lines = debate_summary.split('\n')
            filtered_lines = []
            for line in content_lines:
                # "투자결론리포트" 같은 큰 제목 제거
                if not re.match(r'^#+\s*(투자|결론|리포트)', line, re.IGNORECASE):
                    filtered_lines.append(line)
            st.markdown('\n'.join(filtered_lines))
    else:
        st.info("토론 요약이 생성되지 않았습니다.")


def render_round_by_round_tab(debate_system: DebateSystem):
    """탭 3: 라운드별 의견 변화 렌더링"""
    st.header("🔄 라운드별 의견 변화")
    
    if not debate_system or not debate_system.opinions:
        st.warning("토론 데이터가 없습니다. 먼저 토론을 실행해주세요.")
        return
    
    # 의견 변화 추이 차트
    st.subheader("의견 변화 추이")
    
    rounds = sorted(debate_system.opinions.keys())
    if not rounds:
        st.warning("라운드 데이터가 없습니다.")
        return
    
    # 각 에이전트별 데이터 수집
    agent_names = ["TechnicalAgent", "MacroAgent", "SentimentalAgent"]
    agent_data = {agent: [] for agent in agent_names}
    
    for round_num in rounds:
        opinions = debate_system.opinions.get(round_num, {})
        for agent_id in agent_names:
            opinion = opinions.get(agent_id)
            if opinion and opinion.target:
                agent_data[agent_id].append(opinion.target.next_close)
            else:
                agent_data[agent_id].append(None)
    
    # Ensemble 예측가 (마지막 라운드만)
    ensemble_prices = []
    if st.session_state.ensemble_result:
        ensemble_price = st.session_state.ensemble_result.get("ensemble_next_close")
        for i, round_num in enumerate(rounds):
            if i == len(rounds) - 1 and ensemble_price:
                ensemble_prices.append(ensemble_price)
            else:
                ensemble_prices.append(None)
    
    # 차트 생성
    fig = go.Figure()
    
    colors = {"TechnicalAgent": "#1f77b4", "MacroAgent": "#ff7f0e", "SentimentalAgent": "#2ca02c"}
    
    for agent_id in agent_names:
        if any(agent_data[agent_id]):
            fig.add_trace(go.Scatter(
                x=rounds,
                y=agent_data[agent_id],
                mode="lines+markers",
                name=agent_id,
                line=dict(color=colors.get(agent_id, "#000000"), width=2),
                marker=dict(size=8)
            ))
    
    if any(ensemble_prices):
        fig.add_trace(go.Scatter(
            x=rounds,
            y=ensemble_prices,
            mode="markers",
            name="Ensemble",
            marker=dict(size=12, symbol="star", color="#d62728")
        ))
    
    fig.update_layout(
        title="라운드별 예측가 변화 추이",
        xaxis_title="라운드",
        yaxis_title="예측가 (USD)",
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 라운드별 상세 정보
    st.subheader("라운드별 상세 정보")
    
    round_options = [f"Round {r}" for r in rounds]
    selected_round_str = st.selectbox("라운드 선택", round_options, index=len(round_options)-1)
    selected_round = int(selected_round_str.replace("Round ", ""))
    
    # 선택된 라운드의 의견
    st.write(f"### Round {selected_round} 의견")
    
    round_opinions = debate_system.opinions.get(selected_round, {})
    
    if round_opinions:
        round_opinions_data = []
        for agent_id, opinion in round_opinions.items():
            if opinion and opinion.target:
                round_opinions_data.append({
                    "에이전트": agent_id,
                    "예측가": f"${opinion.target.next_close:,.2f}",
                    "신뢰도": f"{opinion.target.confidence:.4f}" if opinion.target.confidence else "N/A",
                    "불확실성": f"{opinion.target.uncertainty:.4f}" if opinion.target.uncertainty else "N/A",
                    "근거": opinion.reason[:200] + "..." if len(opinion.reason) > 200 else opinion.reason
                })
        
        if round_opinions_data:
            df_round_opinions = pd.DataFrame(round_opinions_data)
            st.dataframe(df_round_opinions, use_container_width=True, hide_index=True)
    else:
        st.info(f"Round {selected_round}의 의견 데이터가 없습니다.")
    
    # 반박/지지 메시지
    if selected_round > 0 and selected_round in debate_system.rebuttals:
        st.write(f"### Round {selected_round} 반박/지지 메시지")
        
        rebuttals = debate_system.rebuttals.get(selected_round, [])
        
        if rebuttals:
            rebuttals_data = []
            for rebut in rebuttals:
                stance_emoji = "❌" if rebut.stance == "REBUT" else "✅"
                rebuttals_data.append({
                    "From": rebut.from_agent_id,
                    "To": rebut.to_agent_id,
                    "Stance": f"{stance_emoji} {rebut.stance}",
                    "Message": rebut.message[:300] + "..." if len(rebut.message) > 300 else rebut.message
                })
            
            df_rebuttals = pd.DataFrame(rebuttals_data)
            st.dataframe(df_rebuttals, use_container_width=True, hide_index=True)
        else:
            st.info(f"Round {selected_round}의 반박/지지 메시지가 없습니다.")
    
    # 반박/지지 패턴 시각화
    if debate_system.rebuttals:
        st.subheader("반박/지지 패턴")
        
        # 라운드별 반박/지지 통계
        pattern_data = []
        for round_num in rounds:
            if round_num > 0:
                rebuttals = debate_system.rebuttals.get(round_num, [])
                rebut_count = sum(1 for r in rebuttals if r.stance == "REBUT")
                support_count = sum(1 for r in rebuttals if r.stance == "SUPPORT")
                pattern_data.append({
                    "라운드": round_num,
                    "반박": rebut_count,
                    "지지": support_count
                })
        
        if pattern_data:
            df_pattern = pd.DataFrame(pattern_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_pattern["라운드"],
                y=df_pattern["반박"],
                name="반박",
                marker_color="#d62728"
            ))
            fig.add_trace(go.Bar(
                x=df_pattern["라운드"],
                y=df_pattern["지지"],
                name="지지",
                marker_color="#2ca02c"
            ))
            
            fig.update_layout(
                title="라운드별 반박/지지 패턴",
                xaxis_title="라운드",
                yaxis_title="개수",
                barmode="group",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)


def main():
    """메인 Streamlit 앱"""
    st.title("AI Stock Debate System")
    st.markdown("다중 에이전트 토론 방식 주식 예측 시스템")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        ticker = st.text_input("티커", value="NVDA", help="분석할 종목 티커를 입력하세요 (예: NVDA, TSLA, AAPL)")
        rounds = st.slider("라운드 수", min_value=1, max_value=5, value=3, help="토론 라운드 수")
        force_pretrain = st.checkbox("Force Pretrain", value=False, help="데이터셋 재생성 및 모델 재학습")
        
        st.divider()
        
        # 토론 시작 버튼
        if st.session_state.is_running:
            st.warning("토론이 실행 중입니다...")
            run_button = st.button("🚀 토론 시작", type="primary", use_container_width=True, disabled=True)
        else:
            run_button = st.button("🚀 토론 시작", type="primary", use_container_width=True)
        
        if run_button and not st.session_state.is_running:
            # 파라미터를 세션 상태에 저장
            st.session_state.run_params = {
                "ticker": ticker.upper(),
                "rounds": rounds,
                "force_pretrain": force_pretrain
            }
            st.session_state.is_running = True
            st.session_state.error_message = None
            st.rerun()
    
    # 토론 실행 (사이드바 밖에서 실행하여 UI 업데이트 가능)
    if st.session_state.is_running and "run_params" in st.session_state:
        try:
            params = st.session_state.run_params
            
            with st.spinner("토론을 실행하는 중... (시간이 걸릴 수 있습니다)"):
                # DebateSystem 초기화 및 실행
                debate_system = DebateSystem(
                    ticker=params["ticker"],
                    rounds=params["rounds"]
                )
                
                ensemble_result = debate_system.run(force_pretrain=params["force_pretrain"])
                
                # 세션 상태에 저장
                st.session_state.debate_system = debate_system
                st.session_state.ensemble_result = ensemble_result
                st.session_state.is_running = False
                del st.session_state.run_params
                
                st.success("토론이 완료되었습니다!")
                st.rerun()
                
        except Exception as e:
            st.session_state.error_message = str(e)
            st.session_state.is_running = False
            if "run_params" in st.session_state:
                del st.session_state.run_params
            st.error(f"오류 발생: {str(e)}")
            with st.expander("상세 오류 정보"):
                st.code(traceback.format_exc())
            st.rerun()
    
    # 에러 메시지 표시
    if st.session_state.error_message:
        st.error(f"오류: {st.session_state.error_message}")
    
    # 메인 컨텐츠 - 탭
    tab1, tab2, tab3 = st.tabs(["📊 기본 주식 데이터", "🎯 최종 결론 및 의견", "🔄 라운드별 의견 변화"])
    
    with tab1:
        # 사이드바에서 입력한 티커 사용 (토론이 실행된 경우 DebateSystem의 티커 우선 사용)
        display_ticker = ticker.upper() if ticker else "NVDA"
        if st.session_state.debate_system:
            display_ticker = st.session_state.debate_system.ticker
        render_stock_overview_tab(display_ticker)
    
    with tab2:
        if st.session_state.debate_system and st.session_state.ensemble_result:
            render_final_conclusion_tab(st.session_state.debate_system, st.session_state.ensemble_result)
        else:
            st.info("토론을 실행한 후 결과를 확인할 수 있습니다.")
    
    with tab3:
        if st.session_state.debate_system:
            render_round_by_round_tab(st.session_state.debate_system)
        else:
            st.info("토론을 실행한 후 결과를 확인할 수 있습니다.")


if __name__ == "__main__":
    main()

