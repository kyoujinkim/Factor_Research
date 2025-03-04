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
import shutil
from glob import glob

import src.factorLib as flib
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

    def turnon_cache(self):
        self.maintain_cache = True

    def turnoff_cache(self):
        self.maintain_cache = False
        del self.cache

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

    def data(self, data_key: str, start: str = None, end: str = None, rolling=0):
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

        data_key = data_key.lower()
        if data_key in self.cache:
            return self.cache[data_key]

        start, end = self.__proc_na_date(start, end)

        if rolling > 0:
            rolling -= 1

        if data_key == 'mktpct':
            self.cache[data_key] = self.__mktpct__(start, end)
        elif data_key == 'aprice':
            self.cache[data_key] = self.__aprice__(start, end)
        elif data_key == 'eps':
            self.cache[data_key] = self.__EPS__(start, end)
        elif data_key == 'epsnorm':
            self.cache[data_key] = self.__EPSnorm__(start, end)
        elif data_key == 'bps':
            self.cache[data_key] = self.__BPS__(start, end)
        elif data_key == 'bpsnorm':
            self.cache[data_key] = self.__BPSnorm__(start, end)
        elif data_key == 'dps':
            self.cache[data_key] = self.__DPS__(start, end)
        elif data_key == 'dpsnorm':
            self.cache[data_key] = self.__DPSnorm__(start, end)
        elif data_key == 'ep':
            self.cache[data_key] = self.__EP__(start, end)
        elif data_key == 'bp':
            self.cache[data_key] = self.__BP__(start, end)
        elif data_key == 'roe':
            self.cache[data_key] = self.__ROE__(start, end)
        elif data_key == 'split':
            self.cache[data_key] = self.__split__(start, end)
        else:
            filelist = glob(f'{BaseData.dbpath}/db/*_{data_key}.parquet')
            filelist = [file for file in filelist if file.split('\\')[-1].split('_')[0] in self.stock_list()]

            query = f"""SELECT date, instrument, """

            if rolling > 0:
                query += f"""SUM({data_key}) OVER (ROWS BETWEEN {rolling} PRECEDING AND 0 FOLLOWING) {data_key} """
            else:
                query += f"""{data_key} """

            query += f"""FROM read_parquet({filelist}, union_by_name=true)
            WHERE date BETWEEN '{start}' AND '{end}'"""

            data = duckdb.sql(query).to_df()

            self.cache[data_key] = data.drop_duplicates(['date', 'instrument'], keep='last').pivot(index='date',
                                                                                                   columns='instrument',
                                                                                                   values=data_key)

        # if we arrived to final stack, delete cache and return value
        if self.cache['depth'] == len(inspect.stack()):
            ret_val = self.cache[data_key]
            if not self.maintain_cache:
                del self.cache

            return ret_val

        return self.cache[data_key]

    def build_data(self
                   , cache_path:str
                   , data:str
                   , start:str=None
                   , end:str=None
                   , data_freq:str='W-FRI'
                   ):
        '''
        build data for weight calculation, since every data saved as parquet, it will be loaded from parquet if exists
        :param start:
        :param end:
        :return: True if success
        '''
        os.makedirs(cache_path, exist_ok=True)

        start_m, end = self.__proc_na_date(start, end, 53)

        if data == 'momentum':
            (flib.momentum(
                self.__cache_data__(cache_path, 'return', start_m, end, data_freq)
                , lt=52, st=4)[start:end]
             .to_parquet(f'{cache_path}/momentum.parquet'))
        elif data == 'investment':
            (flib.investment(
                self.__cache_data__(cache_path, 'totalassets', start_m, end, data_freq)
            )[start:end]
             .to_parquet(f'{cache_path}/investment.parquet'))
        elif data == 'value':
            (flib.value(
                self.__cache_data__(cache_path, 'bp', start_m, end, data_freq)
            )[start:end]
             .to_parquet(f'{cache_path}/value.parquet'))
        elif data == 'size':
            (flib.size(
                self.__cache_data__(cache_path, 'mktcap', start_m, end, data_freq)
            )[start:end]
             .to_parquet(f'{cache_path}/size.parquet'))
        elif data == 'profitability':
            (flib.profitability(
                self.__cache_data__(cache_path, 'roe', start_m, end, data_freq)
            )[start:end]
             .to_parquet(f'{cache_path}/profitability.parquet'))
        else:
            self.__cache_data__(cache_path, data, start_m, end, data_freq)[start:end]

        return True

    def __proc_na_date(self, start:str, end:str, margin_week:int=0):
        if start is None:
            start = '1901-01-01'
        else:
            start = pd.to_datetime(start) - pd.DateOffset(weeks=margin_week)
            start = start.strftime('%Y-%m-%d')
        if end is None:
            end = '2100-12-31'

        return start, end

    def __cache_data__(self
                       , cache_path:str
                       , data:str
                       , start:str
                       , end:str
                       , data_freq:str
                       ):
        if os.path.exists(f'{cache_path}/{data}.parquet'):
            pass
        else:
            if data == 'return':
                (self.data(data, start=start, end=end).resample(data_freq).nearest() / 100).to_parquet(f'{cache_path}/{data}.parquet')
            else:
                self.data(data, start=start, end=end).resample(data_freq).ffill().to_parquet(f'{cache_path}/{data}.parquet')

        return pd.read_parquet(f'{cache_path}/{data}.parquet')

    def __split__(self, start, end):
        filelist = glob(f'{BaseData.dbpath}/db/*_fyshares.parquet')
        filelist = [file for file in filelist if file.split('\\')[-1].split('_')[0] in self.stock_list()]

        query = f"""SELECT date, instrument, """

        query += f"""fyshares / (SELECT fyshares
                                FROM read_parquet({filelist}, union_by_name=true)
                                WHERE date BETWEEN '{start}' AND '{end}'
                                ORDER BY date DESC LIMIT 1) split """

        query += f"""FROM read_parquet({filelist}, union_by_name=true)
        WHERE date BETWEEN '{start}' AND '{end}'"""

        data = duckdb.sql(query).to_df()

        return data.drop_duplicates(['date', 'instrument'], keep='last').pivot(index='date', columns='instrument', values='split')

    def __mktpct__(self, start, end):
        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        mktcap = (self.data('mktcap', start, end) * 1E6).ffill()

        closeprice = self.data('closeprice', start, end).reindex(shares.index, method='ffill').bfill()
        issuecap = shares.multiply(closeprice)

        issuecap.reindex(mktcap.index, method='ffill')

        return issuecap.divide(mktcap)

    def __aprice__(self, start, end):
        closeprice = self.data('closeprice', start, end).ffill()

        split = self.data('split', start, end).ffill()
        split = split.reindex(closeprice.index, method='ffill')
        closeprice = closeprice.multiply(split)

        return closeprice

    def __EPS__(self, start, end):
        earnings = self.data('earnings', start, end, rolling=4).ffill()

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(earnings.index, method='ffill')
        earnings = earnings.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(earnings.index, method='ffill')

        return earnings.divide(shares)

    def __EPSnorm__(self, start, end):
        earnings = self.data('earnings', start, end, rolling=4).ffill()

        split = self.data('split', start, end).ffill()
        split = split.reindex(earnings.index, method='ffill')
        earnings = earnings.multiply(split)

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(earnings.index, method='ffill')
        earnings = earnings.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(earnings.index, method='ffill')

        return earnings.divide(shares)

    def __BPS__(self, start, end):
        bookvalue = self.data('bookvalue', start, end).ffill()

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(bookvalue.index, method='ffill')
        bookvalue = bookvalue.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(bookvalue.index, method='ffill')

        return bookvalue.divide(shares)

    def __BPSnorm__(self, start, end):
        bookvalue = self.data('bookvalue', start, end).ffill()

        split = self.data('split', start, end).ffill()
        split = split.reindex(bookvalue.index, method='ffill')
        bookvalue = bookvalue.multiply(split)

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(bookvalue.index, method='ffill')
        bookvalue = bookvalue.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(bookvalue.index, method='ffill')

        return bookvalue.divide(shares)

    def __DPS__(self, start, end):
        dividend = self.data('dividend', start, end).ffill()

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(dividend.index, method='ffill')
        dividend = dividend.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(dividend.index, method='ffill')

        return dividend.divide(shares)

    def __DPSnorm__(self, start, end):
        dividend = self.data('dividend', start, end).ffill()

        split = self.data('split', start, end).ffill()
        split = split.reindex(dividend.index, method='ffill')
        dividend = dividend.multiply(split)

        mktpct = self.data('mktpct', start, end)
        mktpct = mktpct.reindex(dividend.index, method='ffill')
        dividend = dividend.multiply(mktpct)

        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        shares = shares.reindex(dividend.index, method='ffill')

        return dividend.divide(shares)

    def __EP__(self, start, end):
        aprice = self.data('aprice', start, end).ffill()

        EPS = self.data('EPSnorm', start, end)
        EPS = EPS.reindex(aprice.index, method='ffill')

        return EPS.divide(aprice)

    def __BP__(self, start, end):
        aprice = self.data('aprice', start, end).ffill()

        BPS = self.data('BPSnorm', start, end)
        BPS = BPS.reindex(aprice.index, method='ffill')

        return BPS.divide(aprice)

    def __ROE__(self, start, end):
        EPS = self.data('EPS', start, end)
        BPS = self.data('BPS', start, end)

        EPS = EPS.reindex(BPS.index, method='ffill')

        return EPS.divide(BPS)

if __name__ == '__main__':

    dp = DataProcessor()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'C:/Users/gklb/factor_db/members/master_{index}.parquet', engine='pyarrow')
    curr_index = index_master.iloc[-1:].dropna(axis=1)
    dp.load_stocks(curr_index.columns)

    shutil.rmtree('./cache')
    dp.build_data('./cache', 'momentum', '2020-01-01', '2025-01-01')
    dp.build_data('./cache', 'aprice', '2020-01-01', '2025-01-01')
    a = 1

    a=pd.date_range(start='2000-01-01', end='2025-02-28', freq='w-fri')
    # set a as monthly
    pd.Series(a, index=a).resample('M').nearest()
