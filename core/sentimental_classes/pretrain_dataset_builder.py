import os
import pandas as pd
from core.sentimental_classes.news_history_builder import fetch_history_news
from core.sentimental_classes.finbert_scorer import FinBertScorer
from core.sentimental_classes.news import update_news_db

def build_pretrain_dataset(ticker):
    print(f"[SentimentalAgent] Building pretrain dataset with news for {ticker}...")

    # 🔥 pretrain 디렉토리 자동 생성
    save_dir = "data/pretrain"
    os.makedirs(save_dir, exist_ok=True)

    # 1) 뉴스 수집 (common_params에서 period 가져오기)
    # 주의: pretrain은 FinBERT 스코어를 위해 content가 필요하므로, 
    # update_news_db()와는 별도로 수집합니다.
    # 하지만 run_dataset에서 update_news_db()를 호출하면 증분 업데이트되므로,
    # pretrain 이후에는 중복 수집이 최소화됩니다.
    from datetime import datetime, timedelta
    from config.agents_set import common_params
    period_str = common_params.get("period", "2y")
    # period 문자열을 일수로 변환
    if period_str.endswith("y"):
        years = int(period_str[:-1])
        days = years * 365
    elif period_str.endswith("m"):
        months = int(period_str[:-1])
        days = months * 30
    elif period_str.endswith("d"):
        days = int(period_str[:-1])
    else:
        days = 2 * 365  # 기본값
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    print(f"[SentimentalAgent] 뉴스 수집 기간: {start} ~ {end} ({period_str})")
    print(f"[SentimentalAgent] 참고: pretrain용 뉴스 수집 (FinBERT 스코어 필요). run_dataset에서는 뉴스 DB를 재사용합니다.")
    news_list = fetch_history_news(ticker, start, end)

    # list → DataFrame
    if isinstance(news_list, list):
        df_news = pd.DataFrame(news_list)
    else:
        raise RuntimeError("fetch_history_news did not return list of dict")

    # content 없는 경우 방어
    if "content" not in df_news.columns:
        df_news["content"] = ""

    # 2) FinBERT 스코어
    scorer = FinBertScorer()
    df_news = scorer.score(df_news)

    # 3) 저장
    save_path = f"{save_dir}/{ticker}_news_pretrain.csv"
    df_news.to_csv(save_path, index=False)

    print(f"[SentimentalAgent] Pretrain news saved: {save_path}")
    return df_news
