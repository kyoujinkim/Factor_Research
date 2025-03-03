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

    def __data__(self, data:str
                 , start:str=None
                 , end:str=None
                 ):
        if os.path.exists(f'{self.base_path}/{data}.parquet'):
            pass
        else:
            if data == 'return':
                (self.dp.data(data, start=start, end=end).resample(self.data_freq).nearest() / 100).to_parquet(f'{self.base_path}/{data}.parquet')
            else:
                self.dp.data(data, start=start, end=end).resample(self.data_freq).ffill().to_parquet(f'{self.base_path}/{data}.parquet')

        return pd.read_parquet(f'{self.base_path}/{data}.parquet')


    def build_data(self
                   , data:str
                   , startdate:str=None
                   , enddate:str=None
                   ):
        '''
        build data for weight calculation, since every data saved as parquet, it will be loaded from parquet if exists
        :param startdate:
        :param enddate:
        :return: True if success
        '''
        if data == 'momentum':
            (flib.momentum(self.__data__('return', start=startdate, end=enddate), lt=52, st=4)
             .to_parquet(f'{self.base_path}/momentum.parquet'))
        elif data == 'investment':
            (flib.investment(self.__data__('totalassets', start=startdate, end=enddate))
             .to_parquet(f'{self.base_path}/investment.parquet'))
        else:
            self.__data__(data, start=startdate, end=enddate)

        return True

    def run(self
            , startdate:str=None
            , enddate:str=None
            , weighting_method:str='equal'
            , **kwargs
            ):

        self.dp = ld.DataProcessor()
        self.index = self.index[startdate:enddate].dropna(axis=1, how='all')
        stock_list = self.index.columns
        self.dp.load_stocks(stock_list)
        self.dp.turnon_cache()

        self.base_path = kwargs.get('base_path', './cache')
        factor = kwargs.get('factor', None)
        weight_limit = kwargs.get('weight_limit', None)
        rm = kwargs.get('rm', 'MV')
        workers = kwargs.get('workers', os.cpu_count() - 1)
        active_multiproc = kwargs.get('active_multiproc', True)

        # if factor is designated, build data for factor
        if factor == 'momentum':
            data_list = ['momentum']
        elif factor == 'value':
            data_list = ['bp']
        elif factor == 'size':
            data_list = ['mktcap']
        elif factor == 'investment':
            data_list = ['investment']
        elif factor == 'profitability':
            data_list = ['roe']
        else:
            data_list = []

        if weighting_method == 'mktcap':
            data_list.append('mktcap')
        elif weighting_method == 'optmvo':
            data_list.append('return')
        else:
            pass

        # build data
        for data in data_list:
            self.build_data(data, startdate, enddate)


