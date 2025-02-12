'''
load data 설계
1. index 선택
2. index별로 저장되어있는 index_master 불러오기
    path=D:\Factor_DB\db\master_{index}.csv
3. index_master의 columns에 있는 종목별로 D:/Factor_DB/db에 저장된 종목별 데이터 불러오기
4. 지정된 data를 불러오기 전, data_master를 통해 data 정의
    path=D:\Factor_DB\data_master.json
5. 종목별 data를 불러오기 전, equity_master를 통해 data의 존재 여부, 상태, 업데이트된 기간 확인
    path=D:\Factor_DB\equity_master.json
6. 데이터 로드
'''
import configparser
import inspect
import os
import json
from glob import glob

import pandas as pd

import duckdb


class BaseData:
    """
    Base class for common attributes and methods shared by Stock and Data.
    """
    dbpath = 'D:/Factor_DB'  # Base path for database files

    @staticmethod
    def load_json(file_path: str):
        """
        Utility method to load a JSON file.
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}

    # Load JSON data only once
    data_master = load_json.__func__(f'{dbpath}/data_master.json')
    equity_master = load_json.__func__(f'{dbpath}/equity_master.json')


class Data:
    """
    Class to represent data, inheriting from BaseData.
    """
    def __init__(self, stock:str, data_key: str, start: str, end: str, stat: str, frq: str, cls: str):
        self.stock = stock
        self.data_key = data_key
        self.start = start
        self.end = end
        self.stat = stat
        self.frq = frq
        self.cls = cls

    def __call__(self):
        file_path = f'{BaseData.dbpath}/db/{self.stock}_{self.data_key}.parquet'
        try:
            return pd.read_parquet(file_path, engine='pyarrow')[[self.data_key]].rename(columns={self.data_key: self.stock})
        except FileNotFoundError:
            return f"File not found: {file_path}"

    def __str__(self):
        return f'{self.stock} - {self.data_key}'


def err_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper


class Stock(BaseData):
    """
    Class to represent individual stock, inheriting from BaseData.
    """
    def __init__(self, instrument: str, info: dict):
        super().__init__()
        self.stock = instrument
        self.info = info

        # Dynamically create attributes for each key in the equity_master
        for data_key, data_info in info.items():
            setattr(self, data_key, self.create_data_object(data_key, data_info))

    def __str__(self):
        return f'{self.stock} - {list(self.info.keys())}'

    def create_data_object(self, data_key: str, data_info: dict):
        """
        Create a Data object for the given data key and its associated information.
        """
        if type(data_info)==dict:
            return Data(
                stock=self.stock,
                data_key=data_key,
                start=data_info['start'],
                end=data_info['end'],
                stat=data_info['stat'],
                frq=data_info['frq'],
                cls=BaseData.data_master.get(data_key, {}).get('cls', None)
            )
        else:
            return Data(
                stock=self.stock,
                data_key=data_key,
                start=None,
                end=None,
                stat=None,
                frq=None,
                cls=None
            )


class DataProcessor(BaseData):
    """
    Class to manage data and stocks, inheriting from BaseData.
    """
    def __init__(self):
        super().__init__()
        self.stock = {}

    def data_list(self):
        """
        Return a list of all data.
        """
        return list(self.data_master.keys())

    def stock_list(self):
        """
        Return a list of all stocks.
        """
        return list(self.stock.keys())

    def load_stocks(self, instruments: list):
        """
        Load stocks from the provided instrument list.
        """
        for instrument in instruments:
            stock_info = self.equity_master.get(instrument)
            if stock_info:
                self.stock[instrument] = Stock(instrument, stock_info)
            else:
                print(f"Instrument '{instrument}' not found in equity_master.")

    def unload_stocks(self):
        """
        Unload all stocks.
        """
        self.stock = {}

    def data(self, data_key: str, start: str = None, end: str = None, rolling = 0):
        """
        Load data from the provided data key.
        data_key (str): The key of the data to load.
            - "closepirce" : close price
            - "return" : return
            - "fyshares" : shares
            - "dividend" : dividend
            - "earnings" : earnings
            - "bookvalue" : bookvalue
            - "totalassets" : totalassets
            - "mktcap" : market capitalization
            - "mktpct" : market capitalization percentage
            - "EPS" : earnings per share
            - "EP" : earnings price ratio
            - "BPS" : book value per share
            - "BP" : book value price ratio
            - "ROE" : return on equity
        """
        # set cache for recursive progression
        if not hasattr(self, "cache"):
            setattr(self, "cache", {})
            self.cache['depth'] = len(inspect.stack())
        if data_key in self.cache:
            return self.cache[data_key]

        if start is None:
            start = '1901-01-01'
        if end is None:
            end = '2100-12-31'
        if rolling > 0:
            rolling -= 1

        if data_key == 'mktpct':
            shares = (self.data('fyshares', start, end) * 10E3).ffill()
            mktcap = (self.data('mktcap', start, end) * 10E6).ffill()

            closeprice = self.data('closeprice', start, end).reindex(shares.index, method='ffill').bfill()
            issuecap = shares.multiply(closeprice)

            issuecap = issuecap.reindex(mktcap.index, method='ffill')

            self.cache[data_key] = issuecap.divide(mktcap)

        elif data_key == 'EPS':
            earnings = self.data('earnings', start, end, rolling=4).ffill()

            mktpct = self.data('mktpct', start, end)
            mktpct = mktpct.reindex(earnings.index, method='ffill')
            earnings = earnings.multiply(mktpct)

            shares = (self.data('fyshares', start, end) * 10E3).ffill()
            shares = shares.reindex(earnings.index, method='ffill')

            self.cache[data_key] = earnings.divide(shares)

        elif data_key == 'BPS':
            bookvalue = self.data('bookvalue', start, end).ffill()

            mktpct = self.data('mktpct', start, end)
            mktpct = mktpct.reindex(bookvalue.index, method='ffill')
            bookvalue = bookvalue.multiply(mktpct)

            shares = (self.data('fyshares', start, end) * 10E3).ffill()
            shares = shares.reindex(bookvalue.index, method='ffill')

            self.cache[data_key] = bookvalue.divide(shares)

        elif data_key == 'EP':
            closeprice = self.data('closeprice', start, end).ffill()

            EPS = self.data('EPS', start, end)
            EPS = EPS.reindex(closeprice.index, method='ffill')

            self.cache[data_key] = EPS.divide(closeprice)

        elif data_key == 'BP':
            closeprice = self.data('closeprice', start, end).ffill()

            BPS = self.data('BPS', start, end)
            BPS = BPS.reindex(closeprice.index, method='ffill')

            self.cache[data_key] = BPS.divide(closeprice)

        elif data_key == 'ROE':
            EPS = self.data('EPS', start, end)
            BPS = self.data('BPS', start, end)

            EPS = EPS.reindex(BPS.index, method='ffill')

            self.cache[data_key] = EPS.divide(BPS)

        else:
            filelist = glob(f'{BaseData.dbpath}/db/*_{data_key}.parquet')
            filelist = [file for file in filelist if file.split('\\')[-1].split('_')[0] in self.stock_list()]
            if rolling > 0:
                query = f"""SELECT date, instrument, SUM({data_key}) OVER (ROWS BETWEEN {rolling} PRECEDING AND 0 FOLLOWING) {data_key} 
                FROM read_parquet({filelist}, union_by_name=true)
                WHERE date BETWEEN '{start}' AND '{end}'"""
            else:
                query = f"""SELECT date, instrument, {data_key} 
                FROM read_parquet({filelist}, union_by_name=true)
                WHERE date BETWEEN '{start}' AND '{end}'"""
            data = duckdb.sql(query).to_df()

            self.cache[data_key] = data.drop_duplicates(['date','instrument'], keep='last').pivot(index='date', columns='instrument', values=data_key)

        # if we arrived to final stack, delete cache and return value
        if self.cache['depth'] == len(inspect.stack()):
            ret_val = self.cache[data_key]
            del self.cache
            return ret_val

        return self.cache[data_key]


if __name__ == '__main__':

    dp = DataProcessor()
    dp.load_stocks(['US9898241073'])

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    curr_index = index_master#.iloc[-1:].dropna(axis=1)
    dp.load_stocks(curr_index.columns)

    # check time consumption for dp.data with timeit

    #dp.data('ROE')
    import timeit
    execution_time = timeit.repeat(lambda: dp.data('ROE'), number=1, repeat=5)
    print(f"Execution time: {execution_time} seconds")
    #tmp = dp.data('EP', '2023-01-01')

    a = 1
