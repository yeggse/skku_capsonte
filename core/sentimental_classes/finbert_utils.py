# core/sentimental_classes/finbert_utils.py
# ===============================================================
# FinBERT + 뉴스 캐시 유틸
#  - EODHD 뉴스 수집 + 캐시 저장(load_or_fetch_news)
#  - FinBertScorer (ProsusAI/finbert 기반)
#  - 뉴스별 점수 → 일별 피처 집계 (compute_finbert_features)
# ===============================================================

from __future__ import annotations

from pathlib import Path
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional

import math

ROOT = Path(__file__).resolve().parents[2]  # capstone_project 루트

# ----------------------------------------
# EODHD 뉴스 수집 (필요시 구현)
# ----------------------------------------

def _normalize_symbol(ticker: str) -> str:
    """EODHD 심볼 형식 통일 (NVDA -> NVDA.US)"""
    if not isinstance(ticker, str):
        ticker = str(ticker)
    ticker = ticker.strip().upper()
    if ticker.endswith(".US"):
        return ticker
    return f"{ticker}.US"


def get_news_cache_path(ticker: str, start: date, end: date, news_dir: Optional[Path] = None) -> Path:
    """뉴스 캐시 파일 경로를 한 곳에서만 정의"""
    if news_dir is None:
        news_dir = ROOT / "data" / "raw" / "news"
    else:
        news_dir = Path(news_dir)
    news_dir.mkdir(parents=True, exist_ok=True)

    symbol = _normalize_symbol(ticker)
    filename = f"{symbol}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.json"
    return news_dir / filename


def _fetch_news_from_eodhd_stub(symbol: str, start: date, end: date, api_key: str):
    """
    EODHD에서 뉴스 데이터를 실제로 가져오는 함수.
    EODHDNewsClient를 사용하여 뉴스를 수집하고 적절한 형식으로 변환합니다.
    
    Args:
        symbol: 종목 심볼 (예: "NVDA.US")
        start: 시작 날짜
        end: 종료 날짜
        api_key: EODHD API 키 (없으면 환경변수에서 읽음)
        
    Returns:
        List[Dict[str, Any]]: 뉴스 아이템 리스트
    """
    try:
        from core.sentimental_classes.eodhd_client import EODHDNewsClient
        
        # API 키가 없으면 환경변수에서 읽기
        if not api_key:
            import os
            api_key = os.getenv("EODHD_API_KEY")
        
        if not api_key:
            print("[FinBERT] EODHD_API_KEY가 설정되지 않았습니다. 뉴스 수집을 건너뜁니다.")
            return []
        
        # EODHDNewsClient 초기화 및 뉴스 수집
        client = EODHDNewsClient(api_key=api_key)
        
        # 심볼에서 .US 제거 (EODHDNewsClient가 자동으로 처리할 수도 있음)
        ticker = symbol.replace(".US", "") if symbol.endswith(".US") else symbol
        
        # 날짜 형식 변환 (date -> "YYYY-MM-DD")
        start_str = start.isoformat()
        end_str = end.isoformat()
        
        print(f"[FinBERT] EODHD에서 뉴스 수집 중: {ticker} ({start_str} ~ {end_str})")
        news_items = client.fetch_company_news(
            ticker=ticker,
            from_date=start_str,
            to_date=end_str,
            limit=100  # 최대 100개
        )
        
        # NewsItem을 Dict 형식으로 변환
        result = []
        for item in news_items:
            result.append({
                "date": item.date,
                "published_date": item.date,
                "title": item.title,
                "content": item.content or "",
                "text": item.content or item.title,
                "summary": item.title,
                "tickers": item.tickers or [ticker],
                "source": item.source or "",
                "url": item.url or "",
            })
        
        print(f"[FinBERT] 뉴스 수집 완료: {len(result)}건")
        return result
        
    except ImportError as e:
        print(f"[FinBERT] EODHDNewsClient import 실패: {e}")
        print("[FinBERT] 뉴스 수집을 건너뜁니다.")
        return []
    except Exception as e:
        print(f"[FinBERT] EODHD 뉴스 수집 중 오류 발생: {e}")
        print("[FinBERT] 뉴스 수집을 건너뜁니다.")
        return []


def load_or_fetch_news(
    ticker: str,
    start: date,
    end: date,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    1) 캐시가 있으면 캐시에서 뉴스 로드
    2) 없으면 같은 심볼의 최신 캐시라도 있으면 사용
    3) 그래도 없으면 EODHD에서 가져와 캐시로 저장 (현재는 스텁)
    """
    cache_path = get_news_cache_path(ticker, start, end)
    print(f"[FinBERT] 캐시 탐색: {cache_path} (exists={cache_path.exists()})")

    # 1) 정확히 해당 기간 캐시가 있으면 그대로 사용
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"[FinBERT] 캐시 형식 경고(list 아님): {cache_path}")
        except Exception as e:
            print(f"[FinBERT] 캐시 로드 실패: {cache_path} ({e})")

    # 2) 없으면 폴백: 같은 심볼의 최신 캐시라도 있으면 사용
    news_dir = cache_path.parent
    symbol = _normalize_symbol(ticker)
    candidates = sorted(news_dir.glob(f"{symbol}_*.json"))
    if candidates:
        latest = candidates[-1]
        print(f"[FinBERT] 기존 다른 기간 캐시 사용: {latest.name}")
        try:
            with latest.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                print(f"[FinBERT] 폴백 캐시 형식 경고(list 아님): {latest}")
        except Exception as e:
            print(f"[FinBERT] 폴백 캐시 로드 실패: {latest} ({e})")

    # 3) 그래도 없으면 EODHD 호출 후 저장
    print(f"[FinBERT] 뉴스 캐시 없음: {cache_path}")
    # API 키가 없으면 환경변수에서 읽기
    if not api_key:
        import os
        api_key = os.getenv("EODHD_API_KEY", "")
    data = _fetch_news_from_eodhd_stub(symbol, start, end, api_key=api_key)

    if not isinstance(data, list):
        data = []

    try:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[FinBERT] 뉴스 캐시 저장: {cache_path}")
    except Exception as e:
        print(f"[FinBERT] 뉴스 캐시 저장 실패: {cache_path} ({e})")

    return data


# ----------------------------------------
# FinBERT 모델 로딩 (ProsusAI/finbert)
# ----------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore

    _FINBERT_IMPORT_OK = True
except Exception as e:
    print("[warn] transformers/torch 기반 FinBERT 로딩 실패:", repr(e))
    _FINBERT_IMPORT_OK = False
    torch = None  # type: ignore
    AutoTokenizer = AutoModelForSequenceClassification = None  # type: ignore


class FinBertScorer:
    """
    ProsusAI/finbert 기반 뉴스 감성 스코어러.

    output: 각 텍스트마다
      {
        "pos": 양의 확률,
        "neg": 음의 확률,
        "neu": 중립 확률,
        "score": (pos - neg)  # 대략적인 감성 점수
      }
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        if not _FINBERT_IMPORT_OK:
            raise RuntimeError("transformers/torch 를 불러오지 못해 FinBertScorer 를 생성할 수 없습니다.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
        self.device = device
        self.model.to(self.device)

    def _predict_raw(self, texts: List[str]) -> torch.Tensor:  # type: ignore
        with torch.no_grad():  # type: ignore
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            return torch.softmax(logits, dim=-1).cpu()  # [N, C]

    def score(self, texts: List[str]) -> List[Dict[str, float]]:
        if len(texts) == 0:
            return []

        probs = self._predict_raw(texts)  # [N, 3] (positive/negative/neutral 순서 가정)
        # ProsusAI/finbert 의 label 순서는 보통:
        # id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        pos = probs[:, 0].tolist()
        neg = probs[:, 1].tolist()
        neu = probs[:, 2].tolist()

        out: List[Dict[str, float]] = []
        for p, n, u in zip(pos, neg, neu):
            out.append(
                {
                    "pos": float(p),
                    "neg": float(n),
                    "neu": float(u),
                    "score": float(p - n),
                }
            )
        return out


# ----------------------------------------
# 뉴스 item 리스트 → 점수 부착/집계 유틸
# ----------------------------------------

def score_news_items(
    items: List[Dict[str, Any]],
    scorer: FinBertScorer,
    text_fields: Tuple[str, ...] = ("title", "content", "text", "summary"),
) -> List[Dict[str, float]]:
    texts: List[str] = []
    for it in items:
        parts: List[str] = []
        for f in text_fields:
            v = it.get(f)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        if not parts:
            texts.append("")
        else:
            texts.append("\n".join(parts))
    return scorer.score(texts)


def attach_scores_to_items(
    items: List[Dict[str, Any]],
    scores: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it, sc in zip(items, scores):
        it = dict(it)
        it["finbert"] = sc
        out.append(it)
    return out


def _parse_item_date(it: Dict[str, Any]) -> date | None:
    for key in ("date", "published_date", "time", "pubDate"):
        v = it.get(key)
        if isinstance(v, str) and v:
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(v[:19], fmt).date()
                except Exception:
                    continue
    return None


def compute_finbert_features(
    items_scored: List[Dict[str, Any]],
    asof_utc_date: date,
) -> Dict[str, Any]:
    """
    점수 부착된 뉴스 리스트 → 일별 감성 피처 집계

    반환 예:
      {
        "sentiment_summary": {
          "mean_7d": ...,
          "mean_30d": ...,
          "pos_ratio_7d": ...,
          "neg_ratio_7d": ...,
        },
        "sentiment_volatility": {
          "vol_7d": ...,
        },
        "news_count": {
          "count_1d": ...,
          "count_7d": ...,
        },
        "trend_7d": ...,
      }
    """
    if not items_scored:
        return {
            "sentiment_summary": {
                "mean_7d": 0.0,
                "mean_30d": 0.0,
                "pos_ratio_7d": 0.0,
                "neg_ratio_7d": 0.0,
            },
            "sentiment_volatility": {"vol_7d": 0.0},
            "news_count": {"count_1d": 0, "count_7d": 0},
            "trend_7d": 0.0,
        }

    # 날짜별로 score 모으기
    by_date: Dict[date, List[Dict[str, float]]] = {}
    for it in items_scored:
        d = _parse_item_date(it)
        if d is None:
            continue
        if d not in by_date:
            by_date[d] = []
        sc = it.get("finbert", {})
        if not isinstance(sc, dict):
            continue
        by_date[d].append(sc)

    if not by_date:
        return {
            "sentiment_summary": {
                "mean_7d": 0.0,
                "mean_30d": 0.0,
                "pos_ratio_7d": 0.0,
                "neg_ratio_7d": 0.0,
            },
            "sentiment_volatility": {"vol_7d": 0.0},
            "news_count": {"count_1d": 0, "count_7d": 0},
            "trend_7d": 0.0,
        }

    # asof 기준 30일/7일 윈도우
    dates = sorted(by_date.keys())
    day_scores: Dict[date, float] = {}
    day_pos_ratio: Dict[date, float] = {}
    day_neg_ratio: Dict[date, float] = {}
    day_counts: Dict[date, int] = {}

    for d, lst in by_date.items():
        if not lst:
            continue
        scores = [sc.get("score", 0.0) for sc in lst]
        pos_cnt = sum(1 for sc in lst if sc.get("pos", 0.0) > sc.get("neg", 0.0))
        neg_cnt = sum(1 for sc in lst if sc.get("neg", 0.0) > sc.get("pos", 0.0))
        n = len(lst)
        day_scores[d] = float(sum(scores) / n)
        day_pos_ratio[d] = float(pos_cnt / n) if n > 0 else 0.0
        day_neg_ratio[d] = float(neg_cnt / n) if n > 0 else 0.0
        day_counts[d] = n

    def _window_stats(days: int) -> Tuple[float, float, float, int]:
        start = asof_utc_date - timedelta(days=days)
        vals = [v for d, v in day_scores.items() if start <= d <= asof_utc_date]
        if not vals:
            mean = 0.0
            vol = 0.0
        else:
            mean = float(sum(vals) / len(vals))
            if len(vals) > 1:
                m = mean
                var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
                vol = math.sqrt(var)
            else:
                vol = 0.0

        pos_vals = [day_pos_ratio[d] for d in day_scores if start <= d <= asof_utc_date]
        neg_vals = [day_neg_ratio[d] for d in day_scores if start <= d <= asof_utc_date]
        if pos_vals:
            pos_mean = float(sum(pos_vals) / len(pos_vals))
        else:
            pos_mean = 0.0
        if neg_vals:
            neg_mean = float(sum(neg_vals) / len(neg_vals))
        else:
            neg_mean = 0.0

        cnt = sum(c for d, c in day_counts.items() if start <= d <= asof_utc_date)
        return mean, vol, cnt, pos_mean, neg_mean  # type: ignore

    mean_7d, vol_7d, cnt_7d, pos7, neg7 = _window_stats(7)
    mean_30d, _, _, _, _ = _window_stats(30)

    # 7일 트렌드: 간단한 선형회귀 기울기
    xs: List[float] = []
    ys: List[float] = []
    for d in dates:
        if asof_utc_date - timedelta(days=7) <= d <= asof_utc_date:
            xs.append((d - asof_utc_date).days)
            ys.append(day_scores.get(d, 0.0))
    if len(xs) >= 2:
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        trend = float(num / den) if den != 0 else 0.0
    else:
        trend = 0.0

    # 오늘/어제 뉴스 개수
    count_1d = day_counts.get(asof_utc_date, 0)

    return {
        "sentiment_summary": {
            "mean_7d": float(mean_7d),
            "mean_30d": float(mean_30d),
            "pos_ratio_7d": float(pos7),
            "neg_ratio_7d": float(neg7),
        },
        "sentiment_volatility": {"vol_7d": float(vol_7d)},
        "news_count": {"count_1d": int(count_1d), "count_7d": int(cnt_7d)},
        "trend_7d": float(trend),
    }


__all__ = [
    "_normalize_symbol",
    "get_news_cache_path",
    "load_or_fetch_news",
    "FinBertScorer",
    "score_news_items",
    "attach_scores_to_items",
    "compute_finbert_features",
]
