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
import os
import json
import pandas as pd
import datetime as dt

from pandas._libs.tslibs.offsets import MonthEnd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sqlalchemy import create_engine
from functools import partial


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
        file_path = f'{BaseData.dbpath}/db/{self.stock}_{self.data_key}.csv'
        try:
            return pd.read_csv(file_path, index_col=0, encoding='utf-8-sig', parse_dates=True)
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

    @err_handler
    def mktpct(self):
        '''
        calculate mktpct of issuance
        :return: pd.DataFrame
        '''
        # if mktcap data does not exist, though fyshares data exists, return shape of fyshares dataframe filled with 1
        if self.fyshares.stat is True and self.mktcap.stat is False:
            return pd.DataFrame(1, index=self.fyshares().index, columns=[self.stock])

        shares = self.fyshares() * 10E3
        shares[self.stock] = shares[self.stock].ffill()
        #shares.index = shares.index - MonthEnd(1)
        mktcap = self.mktcap() * 10E6
        mktcap[self.stock] = mktcap[self.stock].ffill()

        shares['closeprice'] = self.closeprice().reindex(shares.index, method='ffill').bfill()
        issuecap = shares[self.stock].multiply(shares['closeprice'])

        issuecap = issuecap.reindex(mktcap.index, method='ffill')

        return pd.DataFrame(issuecap.divide(mktcap[self.stock]))

    @err_handler
    def EPS(self):
        '''
        build EPS data with stocks' earnings and shares data
        Earnings / Shares
        :return: pd.DataFrame
        '''
        earnings = self.earnings()
        shares = self.fyshares() * 10E3

        # get dataframe of date and fydate
        earnings[self.stock] = earnings[self.stock].rolling(4).sum().ffill()
        earnings['mktpct'] = self.mktpct().reindex(earnings.index, method='ffill')
        earnings[self.stock] = earnings[self.stock].multiply(earnings['mktpct'])

        shares[self.stock] = shares[self.stock].ffill()

        shares = shares.reindex(earnings.index, method='ffill')

        return pd.DataFrame(earnings[self.stock].divide(shares[self.stock]))

    @err_handler
    def EP(self):
        '''
        build E/P data with stocks' closeprice and EPS data
        :return:
        '''
        closeprice = self.closeprice().ffill().bfill()
        EPS = self.EPS()

        # extend date index of EPS with closeprice
        EPS = EPS.reindex(closeprice.index, method='ffill')

        return pd.DataFrame(EPS.divide(closeprice))

    @err_handler
    def BPS(self):
        '''
        build BPS data with stocks' bookvalue and shares data
        Bookvalue / Shares
        :return: pd.DataFrame
        '''
        bookvalue = self.bookvalue()
        shares = self.fyshares() * 10E3

        # get dataframe of date and fydate
        bookvalue[self.stock] = bookvalue[self.stock].replace(0, pd.NA).ffill()
        bookvalue['mktpct'] = self.mktpct().reindex(bookvalue.index, method='ffill')
        shares[self.stock] = shares[self.stock].ffill()

        shares = shares.reindex(bookvalue.index, method='ffill')

        return pd.DataFrame(bookvalue[self.stock].divide(shares[self.stock]))

    @err_handler
    def BP(self):
        '''
        build B/P data with stocks' closeprice and BPS data
        :return:
        '''
        closeprice = self.closeprice().ffill().bfill()
        BPS = self.BPS()

        # extend date index of BPS with closeprice
        BPS = BPS.reindex(closeprice.index, method='ffill')

        return pd.DataFrame(BPS.divide(closeprice))

    @err_handler
    def ROE(self):
        '''
        build ROE data with stocks' EPS and BPS data
        EPS / BPS
        :return: pd.DataFrame
        '''
        EPS = self.EPS()
        BPS = self.BPS()

        EPS = EPS.reindex(BPS.index, method='ffill')

        return pd.DataFrame(EPS.divide(BPS))


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
        return list(self.equity_master.keys())

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

    def load_data_dict(self, data_key: str, start: str=None, end: str=None):
        """
        Load data for all stocks for the given data key.
        """
        data = {}
        for stock in tqdm(self.stock.values(), desc=f'Loading {data_key}'):
            data[stock.stock] = getattr(stock, data_key)().loc[start:end]
        return data

    def load_data_df(self, data_key: str, start: str=None, end: str=None):
        """
        Load data for all stocks for the given data key.
        """
        data = []
        for stock in tqdm(self.stock.values(), desc=f'Loading {data_key}'):
            data.append(getattr(stock, data_key)().loc[start:end])

        df = pd.concat(data, axis=1).sort_index()

        return df


if __name__ == '__main__':

    dp = DataProcessor()
    dp.load_stocks(['US9898241073'])
    dp.stock['US9898241073'].EPS()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_csv(f'D:/Factor_DB/db/master_{index}.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    curr_index = index_master.iloc[-1].dropna()
    dp.load_stocks(curr_index.index)
    dp.stock[list(dp.stock)[5]].EPS()

    # import all stock.EPS value and concat
    eps = dp.load_data_df('closeprice')
    #EPS = pd.concat([stock.EPS().loc['2020-01-01':] for stock in dp.stock.values()], axis=1)
    # 3. index_master의 columns에 있는 종목별로 D:/Factor_DB/db에 저장된 종목별 데이터 불러오기
    datalist = ['fyshares', 'closeprice', 'return', 'fyshares', 'dividend', 'earnings', 'bookvalue', 'totalassets']

