# core\sentimental_classes\builders.py
from typing import Dict, Any

def _round(x, n=4):
    try: return None if x is None else round(float(x), n)
    except: return None

def build_prediction_block(stock_data, target):
    last = float(getattr(stock_data, "last_price", 0.0) or 0.0)
    pred_close = float(getattr(target, "next_close", 0.0) or 0.0)
    pred_return = getattr(target, "pred_return", None)
    if pred_return is None and last > 0:
        pred_return = (pred_close - last) / last
    unc = getattr(target, "uncertainty", 0.0) or 0.0
    std = _round(unc.get("std")) if isinstance(unc, dict) else _round(unc)
    ci95 = unc.get("ci95") if isinstance(unc, dict) else None
    pi80 = unc.get("pi80") if isinstance(unc, dict) else None
    conf = _round(getattr(target, "confidence", 0.0), 3)
    prob_up = getattr(target, "calibrated_prob_up", None)
    return {
        "pred_close": _round(pred_close),
        "pred_return": _round(pred_return),
        "uncertainty": {"std": std, "ci95": ci95, "pi80": pi80},
        "confidence": conf,
        "calibrated_prob_up": _round(prob_up, 3) if prob_up is not None else None,
        "mc_mean_next_close": _round(getattr(target, "mc_mean_next_close", None)),
        "mc_std": _round(getattr(target, "mc_std", None)),
    }

# core/sentimental_classes/builders.py

from __future__ import annotations
from typing import List
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

from .eodhd_client import EODHDNewsClient, NewsItem


def build_finbert_news_cache(
    ticker: str,
    from_date: str,
    to_date: str,
    *,
    out_dir: str = "data/raw/news",
    chunk_days: int = 7,
    limit_per_call: int = 100,
    sleep_sec: float = 0.2,
) -> str:
    """
    EODHD API를 이용해 뉴스들을 수집하고,
    SentimentalAgent.build_finbert_news_features()가 읽을 수 있는
    JSON 캐시 파일을 생성한다.

    Parameters
    ----------
    ticker : str
        예) "NVDA"
    from_date : str
        "YYYY-MM-DD" (UTC 기준)
    to_date : str
        "YYYY-MM-DD" (UTC 기준)
    out_dir : str, default "data/raw/news"
        JSON 캐시를 저장할 디렉터리
    chunk_days : int, default 7
        from~to 구간을 며칠 단위로 쪼개서 호출할지
    limit_per_call : int, default 100
        한 번의 API 호출에서 가져올 최대 뉴스 개수
    sleep_sec : float, default 0.2
        API rate-limit을 피하기 위한 호출 간 대기(초)

    Returns
    -------
    str
        생성된 JSON 파일 경로
    """
    import time

    client = EODHDNewsClient()

    start = datetime.fromisoformat(from_date)
    end = datetime.fromisoformat(to_date)

    all_items: List[NewsItem] = []

    cur = start
    while cur <= end:
        cur_to = min(cur + timedelta(days=chunk_days - 1), end)
        s = cur.strftime("%Y-%m-%d")
        e = cur_to.strftime("%Y-%m-%d")
        print(f"[EODHD] fetching {ticker} news {s} ~ {e} ...")

        try:
            items = client.fetch_company_news(
                ticker=ticker,
                from_date=s,
                to_date=e,
                limit=limit_per_call,
            )
            all_items.extend(items)
            print(f"  → {len(items)} 개 수집 (누적 {len(all_items)} 개)")
        except Exception as ex:
            print(f"  !! fetch 실패: {ex}")

        cur = cur_to + timedelta(days=1)
        time.sleep(sleep_sec)

    # 중복 제거 (url 기준, 없으면 날짜+제목 기준)
    seen = set()
    uniq_dicts = []
    for it in all_items:
        key = it.url or f"{it.date}|{it.title}"
        if key in seen:
            continue
        seen.add(key)
        uniq_dicts.append(
            {
                "date": it.date,
                "title": it.title,
                "content": it.content,
                "tickers": it.tickers,
                "source": it.source,
                "url": it.url,
            }
        )

    # 경로 생성 및 저장
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    symbol_us = f"{ticker.upper()}.US"
    json_name = f"{symbol_us}_{from_date}_{to_date}.json"
    json_path = out_path / json_name

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(uniq_dicts, f, ensure_ascii=False, indent=2)

    print(f"✅ 뉴스 캐시 저장 완료: {json_path} (총 {len(uniq_dicts)} 건)")
    return str(json_path)
