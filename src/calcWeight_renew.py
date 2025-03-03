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


