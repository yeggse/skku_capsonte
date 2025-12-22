#!/usr/bin/env python3
"""
Multi-Agent Debate System - Main Entry Point

DebateSystem을 사용하여 여러 에이전트 간의 토론을 실행하고 최종 예측을 생성합니다.

사용법:
    python main.py --ticker NVDA --rounds 3
    python main.py --ticker TSLA --rounds 2 --force-pretrain
"""

import os
import sys
import argparse
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.debate_system import DebateSystem


def main():
    """CLI 인자를 파싱하고 DebateSystem을 초기화하여 토론을 실행합니다."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System - 주식 예측을 위한 다중 에이전트 토론 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (3라운드)
  python main.py --ticker NVDA

  # 라운드 수 지정
  python main.py --ticker TSLA --rounds 5

  # 강제 pretrain 포함 (데이터셋 재생성 및 모델 재학습)
  python main.py --ticker AAPL --rounds 3 --force-pretrain
        """
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="분석할 티커 심볼 (예: NVDA, TSLA, AAPL)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="토론 라운드 수 (기본값: 3)"
    )
    
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="초기 Opinion 수집 시 강제 pretrain 실행 (데이터셋 재생성 포함)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="데이터 디렉토리 (기본값: config에서 가져옴)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="모델 디렉토리 (기본값: config에서 가져옴)"
    )
    
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    print("=" * 80)
    print(f"Multi-Agent Debate System")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Rounds: {args.rounds}")
    print(f"Force Pretrain: {args.force_pretrain}")
    if args.data_dir:
        print(f"Data Dir: {args.data_dir}")
    if args.model_dir:
        print(f"Model Dir: {args.model_dir}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    try:
        debate = DebateSystem(
            ticker=ticker,
            rounds=args.rounds,
            data_dir=args.data_dir,
            model_dir=args.model_dir
        )
        result = debate.run(force_pretrain=args.force_pretrain)
        
        print()
        print("=" * 80)
        print(f"Debate Finished Successfully")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        print("Final Ensemble Result:")
        print("-" * 80)
        
        for key, value in result.items():
            if key == "debate_summary":
                continue
            elif isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            elif key == "agents" and isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.2f}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print("-" * 80)
        
        if "debate_summary" in result and result["debate_summary"]:
            print()
            print("Debate Summary:")
            print("-" * 80)
            print(result["debate_summary"])
            print("-" * 80)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Debate 실행 중 오류 발생:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
