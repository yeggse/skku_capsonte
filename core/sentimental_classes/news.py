import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .news_history_builder import fetch_history_news
from transformers import pipeline
import torch
import time

# 전역 파이프라인 (Lazy Loading)
_sentiment_pipeline = None

def get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        # GPU 사용 가능 시 device=0
        device = 0 if torch.cuda.is_available() else -1
        print(f"[SentimentalAgent] Loading FinBERT pipeline on device {device}...")
        
        # FP16 지원 여부 확인 및 적용
        use_fp16 = False
        if torch.cuda.is_available():
            # CUDA가 있으면 FP16 시도 (대부분의 GPU에서 지원)
            use_fp16 = True
            print(f"[SentimentalAgent] Attempting to use FP16 (Mixed Precision) for faster inference")
        
        # FP16으로 pipeline 생성 시도, 실패 시 FP32로 폴백
        model_kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}
        try:
            _sentiment_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
                device=device,
                tokenizer="ProsusAI/finbert", # 토크나이저 명시
                framework="pt", # PyTorch 프레임워크 명시
                model_kwargs=model_kwargs
            )
            if use_fp16:
                print(f"[SentimentalAgent] FP16 pipeline created successfully")
        except Exception as e:
            if use_fp16:
                print(f"[WARN] FP16 pipeline creation failed: {e}")
                print(f"[SentimentalAgent] Falling back to FP32...")
                _sentiment_pipeline = pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    return_all_scores=True,
                    device=device,
                    tokenizer="ProsusAI/finbert",
                    framework="pt"
                )
            else:
                raise
        
        # 모델 컴파일 (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                print(f"[SentimentalAgent] Compiling model with torch.compile for faster inference...")
                _sentiment_pipeline.model = torch.compile(_sentiment_pipeline.model, mode="reduce-overhead")
                print(f"[SentimentalAgent] Model compilation completed")
            except Exception as e:
                print(f"[WARN] Model compilation failed (continuing without compilation): {e}")
        
    return _sentiment_pipeline

def analyze_sentiment(titles):
    """뉴스 제목 리스트에 대해 감성 점수 계산 (Batch)"""
    pipe = get_pipeline()
    # 배치 크기 설정 (GPU 메모리에 따라 조절)
    # GPU 환경에서는 더 큰 배치 크기로 처리 속도 향상
    batch_size = 128 if torch.cuda.is_available() else 32
    results = pipe(titles, batch_size=batch_size, truncation=True, max_length=64)
    
    scores = []
    labels = []
    
    for res in results:
        # res는 [{'label': 'positive', 'score': 0.9}, ...] 형태
        pos = next(x['score'] for x in res if x['label'] == 'positive')
        neg = next(x['score'] for x in res if x['label'] == 'negative')
        neu = next(x['score'] for x in res if x['label'] == 'neutral')
        
        # 대표 레이블
        max_score = max(pos, neg, neu)
        if max_score == pos: label = 'positive'
        elif max_score == neg: label = 'negative'
        else: label = 'neutral'
        
        # 감성 점수 (Positive - Negative) -> -1 ~ 1
        sentiment_score = pos - neg
        
        scores.append(sentiment_score)
        labels.append(label)
        
    return scores, labels

def update_news_db(ticker, base_dir="data/raw/news", target_start_date=None):
    """
    뉴스 DB 증분 업데이트
    target_start_date: 수집 목표 시작일 (Config의 period 반영). 
                       기존 DB가 없거나, 기존 DB보다 더 과거 데이터가 필요할 때 참조.
    """
    os.makedirs(base_dir, exist_ok=True)
    db_path = os.path.join(base_dir, f"{ticker}_news_db.csv")
    
    df_old = pd.DataFrame()
    start_date = None
    
    # 1. 기존 DB 로드 및 마지막 날짜 확인
    if os.path.exists(db_path):
        try:
            df_old = pd.read_csv(db_path)
            df_old['date'] = pd.to_datetime(df_old['date'])
            
            if not df_old.empty:
                # Timezone 제거 (Naive로 통일)
                if df_old['date'].dt.tz is not None:
                    df_old['date'] = df_old['date'].dt.tz_localize(None)
                
                last_date = df_old['date'].max()
                # 증분 수집: 마지막 날짜 다음 날부터
                start_date = last_date + timedelta(days=1)
                
                # (옵션) 만약 target_start_date가 기존 DB의 시작일보다 훨씬 이전이라면?
                # 현재 구조상 과거 데이터를 prepend하기 어려우므로, 경고만 출력
                current_min_date = df_old['date'].min()
                
                # target_start_date도 Naive로 변환
                ts_date = pd.to_datetime(target_start_date)
                if ts_date.tzinfo is not None:
                    ts_date = ts_date.tz_localize(None)
                    
                if target_start_date and current_min_date > ts_date + timedelta(days=30):
                     print(f"[WARN] {ticker} 기존 뉴스 DB 시작일({current_min_date.date()})이 목표 시작일({target_start_date.date()})보다 늦습니다.")
                     print(f"       과거 데이터가 필요하면 '{db_path}' 파일을 삭제하고 다시 실행하세요.")

        except Exception as e:
            print(f"[WARN] {ticker} 뉴스 DB 로드 실패({e}). 새로 생성합니다.")
            df_old = pd.DataFrame()
    
    # start_date가 설정되지 않았다면 (DB가 없거나 로드 실패, 또는 비어있음)
    if start_date is None:
        if target_start_date:
            start_date = pd.to_datetime(target_start_date)
        else:
            start_date = datetime.now() - timedelta(days=365*2) # 기본값 (폴백)

    # 2. 수집 범위 설정
    end_date = datetime.now()
    
    # 미래 날짜이거나 오늘이면 스킵 (이미 최신)
    if start_date.date() > end_date.date():
        return df_old

    # 3. 뉴스 수집 (Chunking: 30일 단위로 분할 수집)
    str_overall_start = start_date.strftime("%Y-%m-%d")
    str_overall_end = end_date.strftime("%Y-%m-%d")
    print(f"[SentimentalAgent] {ticker} 뉴스 수집 시작: {str_overall_start} ~ {str_overall_end}")

    raw_news = []
    cur_start = start_date
    
    while cur_start <= end_date:
        # 30일 단위 혹은 종료일까지
        cur_end = min(cur_start + timedelta(days=30), end_date)
        
        s_str = cur_start.strftime("%Y-%m-%d")
        e_str = cur_end.strftime("%Y-%m-%d")
        
        try:
            # fetch_history_news는 limit=1000으로 수정됨
            chunk = fetch_history_news(ticker, s_str, e_str)
            if chunk:
                raw_news.extend(chunk)
            time.sleep(0.5) # API Rate Limit 고려
        except Exception as e:
            print(f"[WARN] 뉴스 수집 실패 ({s_str}~{e_str}): {e}")
        
        cur_start = cur_end + timedelta(days=1)

    if not raw_news:
        print(f"[SentimentalAgent] 새로운 뉴스가 없습니다.")
        return df_old

    # 4. 데이터 프레임 변환 및 필터링 (일자별 3개)
    df_new = pd.DataFrame(raw_news)
    
    # 날짜 파싱 (EODHD format: YYYY-MM-DDTHH:MM:SS or similar)
    df_new['date'] = pd.to_datetime(df_new['date'])
    if df_new['date'].dt.tz is not None:
        df_new['date'] = df_new['date'].dt.tz_localize(None)
    df_new['date'] = df_new['date'].dt.normalize() # 시간 제거
    
    # 중복 제거 (동일 날짜, 동일 제목)
    df_new = df_new.drop_duplicates(subset=['date', 'title'])
    
    # 일자별 최대 3개만 남기기
    df_new = df_new.groupby('date').head(3).reset_index(drop=True)
    
    # 제목만 사용
    titles = df_new['title'].tolist()
    print(f"[SentimentalAgent] {len(titles)}개 뉴스 선별 완료 (일자별 최대 3건), 감성 분석 중...")
    
    scores, labels = analyze_sentiment(titles)
    
    df_new['sentiment_score'] = scores
    df_new['sentiment_label'] = labels
    
    # 필요한 컬럼만 유지
    df_new = df_new[['date', 'title', 'sentiment_score', 'sentiment_label']]
    
    # 5. 병합 및 저장
    if not df_old.empty:
        df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['date', 'title'])
    else:
        df_final = df_new
        
    df_final = df_final.sort_values('date').reset_index(drop=True)
    df_final.to_csv(db_path, index=False)
    print(f"[SentimentalAgent] 뉴스 DB 업데이트 완료: {len(df_final)}건")
    
    return df_final

def merge_price_with_news_features(
    df_price: pd.DataFrame,
    ticker: str,
    asof_kst,
    base_dir: str = "data/raw/news",
):
    """
    주가 데이터와 뉴스 통계 데이터를 병합
    """
    
    # df_price의 시작 날짜를 확인하여 뉴스 수집 기간에 반영
    if not df_price.empty:
        # df_price['date']가 datetime인지 확인 (안전장치)
        if 'date' in df_price.columns:
             price_start_date = pd.to_datetime(df_price['date']).min()
        else:
             price_start_date = df_price.index.min()
    else:
        price_start_date = None

    # 1. 뉴스 DB 가져오기 (없으면 생성, 기간은 price_start_date부터)
    df_news = update_news_db(ticker, base_dir, target_start_date=price_start_date)
    
    if df_news.empty:
        print(f"[WARN] {ticker} 뉴스 데이터가 없습니다. 감성 피처를 0으로 채웁니다.")
        df = df_price.copy()
        df["news_count_7d"] = 0
        df["sentiment_mean_7d"] = 0.0
        df["sentiment_vol_7d"] = 0.0
        df["news_count_1d"] = 0
        df["sentiment_mean_1d"] = 0.0
        return df

    # 2. 일별 집계 (Aggregation)
    # 날짜별 그룹화
    daily_stats = df_news.groupby('date').agg(
        count=('sentiment_score', 'count'),
        mean=('sentiment_score', 'mean')
    )
    
    # 3. 파생 변수 생성 (7일 이동평균 등)
    # 날짜 인덱스 채우기 (뉴스가 없는 날은 0)
    idx = pd.date_range(daily_stats.index.min(), daily_stats.index.max())
    daily_stats = daily_stats.reindex(idx, fill_value=0)
    daily_stats.index.name = 'date'
    
    daily_stats['news_count_1d'] = daily_stats['count']
    daily_stats['sentiment_mean_1d'] = daily_stats['mean']
    
    # Rolling 7d
    daily_stats['news_count_7d'] = daily_stats['count'].rolling(7).sum().fillna(0)
    daily_stats['sentiment_mean_7d'] = daily_stats['mean'].rolling(7).mean().fillna(0)
    daily_stats['sentiment_vol_7d'] = daily_stats['mean'].rolling(7).std().fillna(0)
    
    daily_stats = daily_stats.reset_index() # date 컬럼 복원
    
    # 4. 주가 데이터와 병합
    df_price = df_price.copy()
    if 'date' not in df_price.columns:
        df_price['date'] = df_price.index
    
    # df_price의 date는 datetime일 수도 있고 string일 수도 있음. 통일 및 TZ 제거
    df_price['date'] = pd.to_datetime(df_price['date'])
    if df_price['date'].dt.tz is not None:
        df_price['date'] = df_price['date'].dt.tz_localize(None)
        
    # daily_stats의 date도 TZ 제거 (index가 date임에 주의 -> reset_index 했으므로 컬럼)
    if daily_stats['date'].dt.tz is not None:
        daily_stats['date'] = daily_stats['date'].dt.tz_localize(None)
    
    # Left Join
    df_merged = pd.merge(df_price, daily_stats, on='date', how='left')
    
    # 결측치 처리 (뉴스가 없었던 날)
    fill_cols = ['news_count_1d', 'sentiment_mean_1d', 'news_count_7d', 'sentiment_mean_7d', 'sentiment_vol_7d']
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
    
    # 파생 피처 생성 (FEATURE_COLS에 필요한 것들)
    if "return_1d" not in df_merged.columns and "close" in df_merged.columns:
        df_merged["return_1d"] = df_merged["close"].pct_change().fillna(0)
    
    if "hl_range" not in df_merged.columns and all(c in df_merged.columns for c in ["high", "low", "close"]):
        df_merged["hl_range"] = ((df_merged["high"] - df_merged["low"]) / 
                                  df_merged["close"].replace(0, np.nan)).fillna(0)
    
    if "Volume" not in df_merged.columns and "volume" in df_merged.columns:
        df_merged["Volume"] = df_merged["volume"].fillna(0)
            
    return df_merged
