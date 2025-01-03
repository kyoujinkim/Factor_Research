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
        return Data(
            stock=self.stock,
            data_key=data_key,
            start=data_info['start'],
            end=data_info['end'],
            stat=data_info['stat'],
            frq=data_info['frq'],
            cls=BaseData.data_master.get(data_key, {}).get('cls', None)
        )

    def mktpct(self):
        '''
        calculate mktpct of issuance
        :return: pd.DataFrame
        '''
        shares = self.fyshares() * 10E3
        shares[self.stock] = shares[self.stock].ffill()
        #shares.index = shares.index - MonthEnd(1)
        mktcap = self.mktcap() * 10E6
        mktcap[self.stock] = mktcap[self.stock].ffill()

        shares['closeprice'] = self.closeprice().reindex(shares.index, method='ffill').bfill()
        issuecap = shares[self.stock].multiply(shares['closeprice'])

        issuecap = issuecap.reindex(mktcap.index, method='ffill')

        return pd.DataFrame(issuecap.divide(mktcap[self.stock]))


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
        self.stocks = {}

    def load_stocks(self, instruments: list):
        """
        Load stocks from the provided instrument list.
        """
        for instrument in instruments:
            stock_info = self.equity_master.get(instrument)
            if stock_info:
                self.stocks[instrument] = Stock(instrument, stock_info)
            else:
                print(f"Instrument '{instrument}' not found in equity_master.")

    def load_data(self, data_key: str):
        """
        Load data for all stocks for the given data key.
        """
        data = {}
        for stock in self.stocks.values():
            data[stock.stock] = getattr(stock, data_key)()
        return data


def load_col(col_info, data, data_cls):
    col = col_info[0]
    col_end_date = col_info[1]
    if col_end_date == dt.datetime(1990, 1, 1):
        return None
    else:
        col_data = pd.read_csv(f'D:/Factor_DB/db/{col}_{data}.csv', index_col=0, encoding='utf-8-sig')
        if len(col_data) < 1:
            return None
        else:
            #col_data = col_data[~col_data.index.duplicated(keep='first')]
            if data_cls == 'prc':
                col_data = col_data.ffill()
            else:
                col_data = col_data.drop_duplicates(subset='fydate', keep='last')
                col_data = col_data[[col]]
                if data_cls == 'inc':
                    # get 4 data rolling sum
                    col_data = col_data.rolling(4).sum()
                elif data_cls == 'bal':
                    # replace 0 with NaN, and fill NaN with previous value
                    col_data = col_data.replace(0, pd.NA).ffill()

            return col_data


if __name__ == '__main__':

    dp = DataProcessor()
    dp.load_stocks(['US9898241073'])
    dp.stocks['US9898241073'].EPS()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_csv(f'D:/Factor_DB/db/master_{index}.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
    curr_index = index_master.iloc[-1].dropna()
    dp.load_stocks(curr_index.index)
    dp.stocks[list(dp.stocks)[5]].EPS()

    # 3. index_master의 columns에 있는 종목별로 D:/Factor_DB/db에 저장된 종목별 데이터 불러오기
    datalist = ['fyshares', 'closeprice', 'return', 'fyshares', 'dividend', 'earnings', 'bookvalue', 'totalassets']

    for data in datalist:
        # data 성질 규정
        total = []
        data_period = data_master[data]['period']
        data_frq = data_master[data]['frq']
        data_cls = data_master[data]['cls']

        # Create a partial function with fixed arguments
        partial_load_col = partial(load_col, data=data, data_cls=data_cls)

        #create col_data list consisting of [col, end_date, stat]
        col_data = []
        for col in index_master.columns:
            col_data.append([col, equity_master[col][data]['end'], equity_master[col][data]['stat']])

        # multiprocess through index_master.columns
        total = process_map(partial_load_col, col_data, max_workers=os.cpu_count()-1, chunksize=int(len(col_data)/1000))

        # only left dataframe object
        total = [df for df in total if df is not None]
        total = pd.concat(total, axis=1).sort_index()

