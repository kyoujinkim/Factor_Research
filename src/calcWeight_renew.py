import os

import pandas as pd
import src.loadData as ld
import src.factorLib as flib

class WeightBuilder:
    def __init__(self
                 , index:pd.DataFrame
                 , rebal_freq:str='m'
                 , data_freq:str='W-FRI'
                 ):
        '''
        build weight for index
        :param index:
        :param rebal_freq:
        '''
        self.index = index
        self.rebal_freq = rebal_freq
        self.data_freq = data_freq

    def build_data(self
                   , base_path:str
                   , startdate=None
                   , enddate=None
                   ):
        '''
        build data for weight calculation
        :param startdate:
        :param enddate:
        :return: True if success
        '''
        dp = ld.DataProcessor()
        self.index = self.index[startdate:enddate].dropna(axis=1, how='all')
        stock_list = self.index.columns
        dp.load_stocks(stock_list)

        dp.turnon_cache()

        # 1. 데이터 불러오기
        # 1-1. 주간 시세 불러오기
        dp.data('aprice').resample(self.data_freq).ffill().to_parquet(f'{base_path}/price.parquet')
        # 1-2. 시장 정보 불러오기
        (dp.data('return').resample(self.data_freq).nearest() / 100).to_parquet(f'{base_path}/rtn.parquet')
        # 1-4. 시가총액 불러오기
        dp.data('mktcap').resample(self.data_freq).ffill().to_parquet(f'{base_path}/mktval.parquet')
        # 1-5. bookvalue(bps) 불러오기
        dp.data('BP').resample(self.data_freq).ffill().to_parquet(f'{base_path}/bp.parquet')
        # 1-6. earnings(roe) 불러오기
        dp.data('ROE').resample(self.data_freq).ffill().to_parquet(f'{base_path}/roe.parquet')
        # 1-7. totalasset 불러오기
        dp.data('totalassets').resample(self.data_freq).ffill().to_parquet(f'{base_path}/ta.parquet')

        dp.turnoff_cache()

        return True

    def run(self
            , startdate:str=None
            , enddate:str=None
            ):
        base_path = f'./cache'
        if os.path.exists(base_path) == False:
            os.makedirs(base_path)
            self.build_data(base_path, startdate, enddate)
        else:
            pass

