import os
import shutil

import duckdb
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import src.loadData as ld

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
            , start:str=None
            , end:str=None
            , weighting_method:str='equal'
            , **kwargs
            ):
        self.base_path = kwargs.get('base_path', './cache')
        factor = kwargs.get('factor', None)
        weight_limit = kwargs.get('weight_limit', None)
        rm = kwargs.get('rm', 'MV')
        workers = kwargs.get('workers', os.cpu_count() - 1)
        active_multiproc = kwargs.get('active_multiproc', True)

        if start is None:
            start = self.index.index[0]
        start, end = self.__proc_na_date(start, end)
        # get monthly date(last week's Friday), to build monthly portfolio weight
        port_dates = pd.date_range(start=start, end=end, freq=self.data_freq)
        port_dates = pd.Series(port_dates, index=port_dates).resample(self.rebal_freq).last()
        # set list of variable to process in parallel
        # var = [pdate, factor, weighting_method, weight_limit, rm]
        var = [(self.base_path, pdate, self.index.loc[:pdate].iloc[-1].dropna().index, factor, weighting_method, weight_limit, rm) for pdate in port_dates]

        if active_multiproc:
            weights = process_map(self.__calc_weight__, var, max_workers=workers)
        else:
            weights = [self.__calc_weight__(v) for v in var]

        return weights

    @staticmethod
    def __calc_weight__(var):
        '''
        :param var: [pdate, factor, weighting_method, weight_limit, rm]
        :return: weight for pdate
        '''
        path, pdate, code_list, factor, weighting_method, weight_limit, rm = var

        code_list = WeightBuilder.__filter_by_date__(code_list, path, 'return', pdate, 52)
        code_list = WeightBuilder.__filter_by__(code_list, path, 'mktcap', pdate, 100)

        return code_list

    @staticmethod
    def __filter_by_date__(code_list, path:str, data:str, pdate, window:int=52):
        '''
        filter data by date
        :param path:
        :param data:
        :param window:
        :return:
        '''

        sdate = pdate - pd.DateOffset(weeks=window)

        db = WeightBuilder.__load_db__(code_list, path, data, sdate, pdate)
        code_list = db.dropna(axis=1).columns

        return code_list

    @staticmethod
    def __filter_by__(code_list, path:str, data:str, pdate, n:int=-1):
        '''
        filter data by market value
        :param path:
        :param data:
        :param window:
        :return:
        '''
        if n == -1:
            return code_list

        sdate = pdate - pd.DateOffset(weeks=52)

        db = WeightBuilder.__load_db__(code_list, path, data, sdate, pdate, order='DESC', limit=1)
        db = db.dropna(axis=1)

        if len(db) == 0:
            return []
        else:
            code_list = db.sort_values()[:n]
            return

    @staticmethod
    def __load_db__(code_list, path:str, data:str, start:str, end:str, **kwargs):
        '''
        load data from db
        :param path:
        :param data:
        :param start:
        :param end:
        :return:
        '''
        order = kwargs.get('order', 'ASC')
        limit = kwargs.get('limit', 'NULL')

        line_query = f"SELECT * FROM '{path}/{data}.parquet' LIMIT 1"
        line_data = duckdb.sql(line_query).to_df()

        valid_columns = [col for col in code_list if col in line_data.columns]

        query = f"""SELECT {', '.join(valid_columns)}, date
        FROM '{path}/{data}.parquet'
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date {order}
        LIMIT {limit}
        """
        result = duckdb.sql(query).to_df().set_index('date')

        return result

    @staticmethod
    def __proc_na_date(start:str, end:str, margin_week:int=0):
        if start is None:
            start = '2000-01-01'
        else:
            start = pd.to_datetime(start) - pd.DateOffset(weeks=margin_week)
            start = start.strftime('%Y-%m-%d')
        if end is None:
            today = pd.Timestamp.today()
            end = today.strftime('%Y-%m-%d')

        return start, end

if __name__ == '__main__':

    factor = 'momentum'
    weighting_method = 'equal'

    dp = ld.DataProcessor()
    dp.turnon_cache()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'C:/Users/gklb/factor_db/members/master_{index}.parquet', engine='pyarrow')
    #curr_index = index_master.iloc[-1:].dropna(axis=1)
    dp.load_stocks(index_master.columns)

    bpath = './cache/us'
    shutil.rmtree(bpath, ignore_errors=True)
    os.makedirs(bpath, exist_ok=True)

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

    data_list.append('mktcap')

    # build data
    for data in tqdm(data_list):
        dp.build_data(bpath, data, None, None)

    wb = WeightBuilder(index_master, rebal_freq='m', data_freq='W-FRI')
    wb.run(start=None, end=None, weighting_method=weighting_method, factor=factor, base_path=bpath, active_multiproc=False)