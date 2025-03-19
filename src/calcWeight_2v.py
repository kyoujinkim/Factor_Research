import os
import pickle
import shutil
from glob import glob

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from multiprocessing.shared_memory import SharedMemory
import src.loadData as ld

# Disable DuckDB's progress bar
duckdb.sql("PRAGMA enable_progress_bar=false")

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
            , **kwargs
            ):
        workers = kwargs.get('workers', os.cpu_count() - 1)
        active_multiproc = kwargs.get('active_multiproc', True)

        if start is None:
            start = self.index.index[0]
        start, end = self.__proc_na_date(start, end)
        # get monthly date(last week's Friday), to build monthly portfolio weight
        port_dates = pd.date_range(start=start, end=end, freq=self.data_freq)
        port_dates = pd.Series(port_dates, index=port_dates).resample(self.rebal_freq).last()

        data_list = glob(f"{kwargs.get('base_path', './cache')}/*.parquet")
        for data in data_list:
            self.shmem(data) # Corrected syntax

        # set list of variable to process in parallel
        var = [
            {
            'pdate':pdate
            , 'code_list':self.index.loc[:pdate].iloc[-1].dropna().index
            , **kwargs
            }
            for pdate in port_dates]

        if active_multiproc:
            weights = process_map(WeightBuilder.__calc_weight__, var, max_workers=workers)
        else:
            weights = [WeightBuilder.__calc_weight__(v) for v in var]

        return pd.concat(weights,axis=0)

    def shmem(self, path):
        datadf = pd.read_parquet(path, engine='pyarrow')
        datapk = pickle.dumps(datadf)
        # Create shared memory
        shm = SharedMemory(create=True, size=len(datapk))
        setattr(self, path, shm)

        # Create a NumPy array from shared memory buffer
        buf_array = np.frombuffer(shm.buf, dtype=np.uint8)
        setattr(self, path + '_buf', buf_array)

        # Copy serialized data into shared memory
        buf_array[:] = np.frombuffer(datapk, dtype=np.uint8)

    @staticmethod
    def __load_mem__(name):
        # Retrieve data from shared memory (for validation)
        shm = SharedMemory(name=name)  # Attach to existing shared memory
        datapkex = bytes(shm.buf[:])  # Read bytes
        datadfex = pickle.loads(datapkex)
        # Close shared memory
        shm.close()  # Corrected syntax

        return datadfex

    @staticmethod
    def __calc_weight__(var):
        '''
        :param var: [path, pdate, factor, weighting_method, weight_limit, rm]
        :return: weight for pdate
        '''
        path = var.get('base_path', './cache')
        pdate = var.get('pdate', None)
        code_list = var.get('code_list', [])
        factor = var.get('factor', None)
        mktcap_n = var.get('mktcap_n', -1)
        factor_n = var.get('factor_n', -1)
        weighting_method = var.get('weighting_method', 'equal')
        weight_limit = var.get('weight_limit', None)

        if len(code_list)>0:
            code_list, _ = WeightBuilder.__filter_by_date__(code_list, path, 'return', pdate, 52)
        if len(code_list)>0:
            code_list, mktcap_db = WeightBuilder.__filter_by__(code_list, path, 'mktcap', pdate, mktcap_n)
        if len(code_list)>0 and factor:
            code_list, factor_db = WeightBuilder.__filter_by__(code_list, path, factor, pdate, factor_n)

        if len(code_list)==0:
            w = pd.DataFrame(columns=[pdate])

        if weighting_method == 'equal':
            w = WeightBuilder.__weight_by__(code_list, 'equal')
        elif weighting_method == 'mktcap':
            w = WeightBuilder.__weight_by__(code_list, 'weight', mktcap_db)
        elif weighting_method == 'factor':
            w = WeightBuilder.__weight_by__(code_list, 'weight', factor_db)

        w.columns = [pdate]

        return w.T

    @staticmethod
    def __weight_by__(code_list, method, db=None):
        if method == 'equal':
            w = pd.DataFrame(1 / len(code_list), index=code_list, columns=['weight'])
        elif method == 'weight':
            w = db[code_list]
            w = w / w.sum()

        return w.fillna(0)

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
        code_list = db.dropna(axis=1).columns.to_list()

        return code_list, db

    @staticmethod
    def __filter_by__(code_list, path:str, data:str, pdate, n:int=-1):
        '''
        filter data by market value
        :param path:
        :param data:
        :param window:
        :return:
        '''

        sdate = pdate - pd.DateOffset(weeks=4)
        db = WeightBuilder.__load_db__(code_list, path, data, sdate, pdate, order='DESC', limit=1)
        db = db.dropna(axis=1)

        if len(db) == 0:
            return [], None
        else:
            code_list = WeightBuilder.__filter_n__(n, db)
            code_list = code_list.columns.to_list()
            return code_list, db

    @staticmethod
    def __filter_n__(n, db):
        if type(n) == list:
            # if top_n is int, calculate top and bottom
            if type(n[0]) == int:
                top = n[0]
                bottom = n[1]
            # if top_n is float, calculate top and bottom
            elif type(n[0]) == float:
                top = min(int(n[0] * len(db)), len(db) / 2)
                bottom = min(int(n[1] * len(db)), len(db) / 2)
            else:
                raise ValueError('element of top_n(top, bottom) must be int or float')
            code_list = db.sort_values()[top:bottom]
        elif type(n) == int:
            if n == -1:
                code_list = db
            else:
                code_list = db.sort_values()[:n]
        elif type(n) == float:
            top = int(n * len(db))
            code_list = db.sort_values()[:top]
        else:
            raise ValueError('top n must be int or float or list')

        return code_list

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

        file_path = f"{path}/{data}.parquet"

        # Load file into shared memory if not already loaded
        if file_path not in WeightBuilder.shared_mem_managers:
            WeightBuilder.shared_mem_managers[file_path] = smm(file_path)
            WeightBuilder.shared_mem_managers[file_path].load_to_shared_memory()

        # Read data from shared memory
        df = WeightBuilder.shared_mem_managers[file_path].read_from_shared_memory()

        # Filter by date range
        df = df.loc[start:end]

        valid_columns = [col for col in code_list if col in df.columns]
        df = df[valid_columns]

        return df

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
    #shutil.rmtree(bpath, ignore_errors=True)
    #os.makedirs(bpath, exist_ok=True)

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
    #for data in tqdm(data_list):
    #    dp.build_data(bpath, data, None, None)

    wb = WeightBuilder(index_master, rebal_freq='m', data_freq='W-FRI')
    w = wb.run(start='2024-06-30'
               , end=None
               , weighting_method=weighting_method
               , factor=factor
               , base_path=bpath
               , workers=3
               , active_multiproc=True)

    w.to_csv('tmp.csv')