from config.agents_set import dir_info
from core.macro_classes.macro_class_dataset import MacroAData
'''
티커 통합 모델
'''

# 매크로 데이터셋 생성 함수 (build_dataset 기능)
# 모델도 함께 생성됨
def macro_dataset(ticker_name):
    print(f"[TRACE B] macro_dataset() start for {ticker_name}")
    macro_data_agent = MacroAData(ticker_name)
    macro_data_agent.fetch_data()
    macro_data_agent.add_features()
    macro_data_agent.save_csv()
    macro_data_agent.make_close_price()
    print(f"macro: 데이터셋 생성 완료> {ticker_name}")

    macro_data_agent.model_maker()
    print(f"macro: 모델 생성 완료> {ticker_name}")


