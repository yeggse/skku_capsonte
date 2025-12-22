# core\sentimental_classes\eodhd_client.py
from __future__ import annotations
import os
import time
import typing as T
import requests
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

@dataclass
class NewsItem:
    date: str
    title: str
    content: T.Optional[str]
    tickers: T.List[str]
    source: T.Optional[str]
    url: T.Optional[str]

class EODHDNewsClient:
    """
    EODHD 뉴스만 사용해서 수집.
    - 기본 endpoint, 파라미터는 .env에 둠 (변경 용이)
    - 최소 의존성: requests, python-dotenv
    """
    def __init__(self, api_key: T.Optional[str] = None, base_url: T.Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        self.base_url = (base_url or os.getenv("EODHD_BASE_URL") or "https://eodhd.com").rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("EODHD_API_KEY 가 설정되어 있지 않습니다 (.env 또는 환경변수 확인).")

        # EODHD 뉴스 엔드포인트 경로를 한 곳에 모아둠 (문서 변경 시 여기만 바꾸면 됨)
        self.news_path = "/api/news"

    def _get(self, path: str, params: dict) -> T.Any:
        url = f"{self.base_url}{path}"
        # 공통 파라미터
        params = {**params, "api_token": self.api_key, "fmt": "json"}
        r = requests.get(url, params=params, timeout=self.timeout)
        if r.status_code == 401 or r.status_code == 403:
            raise RuntimeError(f"EODHD 인증 오류(status={r.status_code}). API 키/플랜 확인 필요: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"EODHD 에러(status={r.status_code}): {r.text[:200]}")
        try:
            return r.json()
        except Exception as e:
            raise RuntimeError(f"JSON 파싱 실패: {e}; raw={r.text[:200]}")

    def fetch_company_news(
        self,
        ticker: str,
        from_date: T.Optional[str] = None,  # "YYYY-MM-DD"
        to_date: T.Optional[str]   = None,  # "YYYY-MM-DD"
        limit: int = 50
    ) -> T.List[NewsItem]:
        """
        ticker별 뉴스 수집. from_date/to_date 없으면 최근 기준 기본 제공(엔드포인트 기본 동작에 따름).
        """
        params = {"s": ticker, "limit": limit}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._get(self.news_path, params)
        # EODHD 응답 구조를 관용적으로 파싱 (필드명은 계정/플랜/엔드포인트에 따라 달라질 수 있어 넉넉히 처리)
        items: T.List[NewsItem] = []
        if isinstance(data, list):
            for x in data:
                items.append(
                    NewsItem(
                        date=str(x.get("date") or x.get("publishedDate") or ""),
                        title=str(x.get("title") or ""),
                        content=x.get("content") or x.get("text") or None,
                        tickers=x.get("tickers") or x.get("symbols") or [],
                        source=x.get("source") or x.get("source_name") or None,
                        url=x.get("url") or x.get("link") or None,
                    )
                )
        elif isinstance(data, dict) and "news" in data and isinstance(data["news"], list):
            for x in data["news"]:
                items.append(
                    NewsItem(
                        date=str(x.get("date") or x.get("publishedDate") or ""),
                        title=str(x.get("title") or ""),
                        content=x.get("content") or x.get("text") or None,
                        tickers=x.get("tickers") or x.get("symbols") or [],
                        source=x.get("source") or x.get("source_name") or None,
                        url=x.get("url") or x.get("link") or None,
                    )
                )
        else:
            # 구조가 전혀 다르면 원문 일부를 보여주도록
            raise RuntimeError(f"예상과 다른 응답 구조: {str(data)[:300]}")

        return items

    def save_to_csv(self, items, ticker: str, out_dir: str = "data/news") -> str:
        import pandas as pd, os
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ticker.upper()}_news.csv")

        # 신규 데이터 → DF
        new_df = pd.DataFrame([{
            "date": it.date,
            "title": it.title,
            "content": it.content or "",
            "tickers": "|".join(it.tickers or []),
            "source": it.source or "",
            "url": it.url or "",
        } for it in items], columns=["date","title","content","tickers","source","url"])

        # 기존 파일 로드(없으면 빈 DF)
        if os.path.exists(out_path):
            try:
                old_df = pd.read_csv(out_path, encoding="utf-8-sig")
            except Exception:
                old_df = pd.DataFrame(columns=["date","title","content","tickers","source","url"])
        else:
            old_df = pd.DataFrame(columns=["date","title","content","tickers","source","url"])

        # 기존+신규 합치고 url 기준 dedup
        all_df = pd.concat([old_df, new_df], ignore_index=True)
        if "url" not in all_df.columns:
            all_df["url"] = ""
        all_df = all_df.drop_duplicates(subset=["url"], keep="first")

        # 날짜 정렬 시도(가능하면)
        try:
            _dt = pd.to_datetime(all_df["date"], errors="coerce", utc=True)
            all_df = all_df.loc[_dt.sort_values().index]
        except Exception:
            pass

        # 전체 저장
        all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out_path

