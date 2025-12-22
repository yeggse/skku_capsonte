import os
import json
import time
import requests
from requests.exceptions import Timeout, RequestException
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# API KEY 로드
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise RuntimeError(
        "EODHD_API_KEY is missing. Add it to your .env file: EODHD_API_KEY=xxx"
    )

def fetch_history_news(ticker, start, end, max_retries=3, timeout=60):
    """
    EODHD API에서 뉴스 데이터를 가져옵니다.
    
    Args:
        ticker: 종목 코드
        start: 시작 날짜 (YYYY-MM-DD)
        end: 종료 날짜 (YYYY-MM-DD)
        max_retries: 최대 재시도 횟수 (기본값: 3)
        timeout: 타임아웃 시간(초) (기본값: 60)
    
    Returns:
        뉴스 데이터 리스트
    """
    url = (
        f"https://eodhd.com/api/news?"
        f"s={ticker}&from={start}&to={end}&limit=1000&api_token={EODHD_API_KEY}&fmt=json"
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            
            # 요청 실패
            if not r.ok:
                raise RuntimeError(f"[fetch_history_news] HTTP {r.status_code}: {r.text}")

            # JSON parse
            try:
                data = r.json()
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"[fetch_history_news] Invalid JSON response:\n{r.text[:300]}"
                )

            # API 오류 메시지
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"[fetch_history_news] API Error: {data}")

            return data
            
        except Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 2초, 4초, 6초...
                print(f"[fetch_history_news] 타임아웃 발생 (시도 {attempt + 1}/{max_retries}). {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                print(f"[fetch_history_news] 최대 재시도 횟수({max_retries}) 초과. 타임아웃 발생.")
                raise RuntimeError(f"[fetch_history_news] 타임아웃: {e}")
                
        except RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[fetch_history_news] 요청 오류 발생 (시도 {attempt + 1}/{max_retries}): {e}. {wait_time}초 후 재시도...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"[fetch_history_news] 요청 실패: {e}")
    
    # 이 코드는 도달하지 않아야 하지만 안전장치
    raise RuntimeError(f"[fetch_history_news] 예상치 못한 오류: {last_error}")
