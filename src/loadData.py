'''
load data 설계
DB 경로는 config.DB_PATH 한 곳에서만 지정한다 (BaseData.dbpath가 그 값을 들고 있고, 다른 모듈은 이것을 참조).
1. index 선택
2. index별로 저장되어있는 index_master 불러오기
    path={dbpath}/members/master_{index}.parquet
3. index_master의 columns에 있는 종목별로 {dbpath}/db에 저장된 종목별 데이터 불러오기
4. 지정된 data를 불러오기 전, data_master를 통해 data 정의
    path={dbpath}/data_master.json
5. 종목별 data를 불러오기 전, equity_master를 통해 data의 존재 여부, 상태, 업데이트된 기간 확인
    path={dbpath}/equity_master.db
6. 데이터 로드
'''
import inspect
import os
import json
import shutil
from glob import glob

import src.factorLib as flib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import duckdb

try:
    from config import DB_PATH
except ImportError:     # run from outside the repo root
    DB_PATH = r'C:\Users\NHWM\Factor_DB'
from src.data_manager import MasterDataManager
from src.rawstore import RawStore, StockFiles

# build_data's cache files carry a signature in their parquet metadata; see DataProcessor.__cache_data__
CACHE_SIG_KEY = b'factor_research.cache'
CACHE_VERSION = 3     # 3: look-ahead fix, TTM per stock, line-level amktcap + cmktcap
# set when every raw date already lies on the resampling grid: the file then gives the same as-of value as the raw
# data at any date, so it can stand in for it (used for zprice inside __zDPS__)
CACHE_ON_GRID_KEY = b'factor_research.on_grid'

FULL_START = '1901-01-01'
# Sparse (quarterly / irregular) level variables stay valid until the next report, which can be older than any fixed
# look-back margin, so data() read from a start date also takes each stock's last value before it.
# Not included: flows (returns, dividend, earnings), which must never be carried forward, and the weekly level
# variables (closeprice, fyshares, mktcap), which build_data's 53-week margin always covers for held stocks (checked
# for all five markets) and whose carry-in would widen every frame with delisted stocks and change __mktpct__,
# which reads closeprice without ffill.
LEVEL_KEYS = {'bookvalue', 'totalassets', 'freefloatpct'}


class BaseData:
    """
    Base class for common attributes and methods shared by Stock and Data.
    """
    dbpath = DB_PATH  # Base path for database files

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

    @staticmethod
    def load_db(file_path: str):
        """
        Utility method to load a database file.
        """
        if os.path.exists(file_path):
            return MasterDataManager(file_path)
        else:
            raise FileNotFoundError('equity_master.db file not found')

    # Load JSON data only once
    data_master = load_json.__func__(f'{dbpath}/data_master.json')
    equity_master = load_db.__func__(f'{dbpath}/equity_master.db')


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
        self.files = StockFiles(f'{BaseData.dbpath}/db')    # the one listing of the per-stock files for this instance
        self.raw_store = None
        self._source_sig = None
        self._zprice_weekly = None      # (end, weekly full-history zprice) loaded by build_data, see _full_zprice
        self.turnoff_cache()

    def set_raw_store(self, root: str = None):
        """
        Read raw data through the consolidated store (see src/rawstore.py) instead of scanning every per-stock file.
        The per-stock files stay the source of truth; the store is rebuilt/patched from them on read.
        """
        self.raw_store = RawStore(root or f'{BaseData.dbpath}/derived', f'{BaseData.dbpath}/db', files=self.files)

    def turnon_cache(self):
        self.maintain_cache = True

    def turnoff_cache(self):
        self.maintain_cache = False
        try:
            del self.cache
        except:
            pass

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
            stock_info = self.equity_master.get_section(instrument)
            if stock_info:
                self.stock[instrument] = Stock(instrument, stock_info)
            else:
                print(f"Instrument '{instrument}' not found in equity_master.")
        self._source_sig = None
        self._zprice_weekly = None

    def unload_stocks(self):
        """
        Unload all stocks.
        """
        self.stock = {}
        self._source_sig = None
        self._zprice_weekly = None

    def tag(self, data_key:str):
        """
        Load tag from the provided data key.
        data_key (str): The key of the data to load.
            - "name" : name
            - "sector" : sector
            - "issuename" : issue name
            - "country" : country
            - "currency" : currency
        """
        data_key = data_key.lower()
        result = []
        if data_key in self.data_master:
            if self.data_master[data_key]['cls'] == 'tag':
                for s in self.stock.keys():
                   result.append([s, self.stock[s].info.get(data_key, None)])


                return pd.DataFrame(result, columns=pd.Index(['instrument', data_key]), dtype=str).set_index('instrument')
            else:
                raise ValueError(f"Data key '{data_key}' is not a tag.")
        else:
            raise ValueError(f"Data key '{data_key}' not found in data_master.")

    def data(self, data_key: str, start: str = None, end: str = None, fy: bool = False, rolling=0):
        """
        Load data from the provided data key.
        data_key (str): The key of the data to load.
            - "closepirce" : close price
            - "returns" : return
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
        start, end = self.__proc_na_date(start, end)
        # the same name with other arguments is a different result, so the arguments are part of the key
        ckey = (data_key, start, end, fy, rolling)
        if ckey in self.cache:
            return self.cache[ckey]

        if rolling > 0:
            rolling -= 1

        derived = {
            'mktpct': self.__mktpct__, 'amktcap': self.__amktcap__, 'cmktcap': self.__cmktcap__, 'aprice': self.__aprice__, 'zprice': self.__zprice__,
            'eps': self.__EPS__, 'epsnorm': self.__EPSnorm__, 'bps': self.__BPS__, 'bpsnorm': self.__BPSnorm__,
            'dps': self.__DPS__, 'dpsnorm': self.__DPSnorm__, 'zdps': self.__zDPS__,
            'ep': self.__EP__, 'bp': self.__BP__, 'roe': self.__ROE__, 'split': self.__split__,
        }
        if data_key in derived:
            self.cache[ckey] = derived[data_key](start, end)
        else:
            # a level variable read from a start date also gets each stock's last value before it (see _seed_start)
            seed = start > FULL_START and data_key in LEVEL_KEYS and rolling == 0 and not fy
            source, raw_rows, row_order = None, None, None
            if self.raw_store is not None:
                raw = self.raw_store.ensure(data_key, self.stock)
                if raw is not None:
                    raw_rows = self.raw_store.read(data_key, raw[1])
                    if seed:
                        raw_rows = raw_rows.append_column('__row', pa.array(np.arange(raw_rows.num_rows)))
                        row_order = '__row'
                    duckdb.register('raw_store_rows', raw_rows)
                    source = 'raw_store_rows'

            if source is None:
                filelist = glob(f'{BaseData.dbpath}/db/*_{data_key}.parquet')
                stocks = set(self.stock)
                filelist = [file for file in filelist if file.split('\\')[-1].split('_')[0] in stocks]
                source = f"read_parquet({filelist}, union_by_name=true{', file_row_number=true' if seed else ''})"
                row_order = 'file_row_number'     # one file per stock, so this is the row order within a stock

            colsubset = 'fydate' if fy else data_key
            cols = 'date, fydate, instrument' if fy else 'date, instrument'

            if rolling > 0:
                # trailing sum over this stock's own last rolling+1 reports, only when all of them have a value. The
                # window runs before the start filter, so the first rows of a window starting at `start` are complete
                # (a result read from a start date equals the full-history one)
                query = f"""SELECT {cols}, {data_key} FROM (
                    SELECT {cols},
                           CASE WHEN COUNT({data_key}) OVER w = {rolling + 1} THEN SUM({data_key}) OVER w END AS {data_key}
                    FROM {source}
                    WHERE date <= '{end}'
                    WINDOW w AS (PARTITION BY instrument ORDER BY date ROWS BETWEEN {rolling} PRECEDING AND CURRENT ROW))
                WHERE date BETWEEN '{start}' AND '{end}'"""
            else:
                query = f"""SELECT {cols}, {data_key} FROM {source}
                WHERE date BETWEEN '{start}' AND '{end}'"""

            try:
                data = duckdb.sql(query).to_df()
                seeds = self._last_before(source, data_key, start, row_order) if seed else None
            finally:
                if raw_rows is not None:
                    duckdb.unregister('raw_store_rows')

            wide = self._pivot_last(data, 'date', 'instrument', colsubset)
            self.cache[ckey] = self._seed_start(wide, seeds, start) if seed else wide

        # if we arrived to final stack, delete cache and return value
        if self.cache['depth'] == len(inspect.stack()):
            ret_val = self.cache[ckey]
            if not self.maintain_cache:
                del self.cache

            return ret_val

        return self.cache[ckey]

    @staticmethod
    def _last_before(source: str, key: str, start: str, row_order: str) -> pd.Series:
        """
        Per stock, the value a full-history ffill would carry into `start`: the latest date before `start` whose
        value is valid, where a date's value is its last row (the same rule as _pivot_last).
        """
        return duckdb.sql(f"""
            WITH per_date AS (
                SELECT instrument, date, {key} FROM {source}
                WHERE date < '{start}'
                QUALIFY row_number() OVER (PARTITION BY instrument, date ORDER BY {row_order} DESC) = 1)
            SELECT instrument, {key} FROM per_date
            WHERE {key} IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY instrument ORDER BY date DESC) = 1
        """).to_df().set_index('instrument')[key]

    @staticmethod
    def _seed_start(wide: pd.DataFrame, seeds: pd.Series, start: str) -> pd.DataFrame:
        """put the carried-in values on the `start` row, only where that row has no value of its own"""
        if seeds is None or len(seeds) == 0:
            return wide
        ts = pd.Timestamp(start)
        index = wide.index.union(pd.DatetimeIndex([ts])); index.name = wide.index.name or 'date'
        columns = wide.columns.union(seeds.index); columns.name = wide.columns.name or 'instrument'
        wide = wide.reindex(index=index, columns=columns)
        wide.loc[ts] = wide.loc[ts].fillna(seeds.reindex(columns))
        return wide

    @staticmethod
    def _pivot_last(df: pd.DataFrame, index: str, columns: str, values: str):
        """
        Same result as df.drop_duplicates([index, columns], keep='last').pivot(index=index, columns=columns, values=values),
        but fills the wide array directly (later rows overwrite earlier ones).
        """
        if df[values].dtype != np.float64:
            return df.drop_duplicates([index, columns], keep='last').pivot(index=index, columns=columns, values=values)

        row_codes, rows = pd.factorize(df[index], sort=True)
        col_codes, cols = pd.factorize(df[columns], sort=True)
        ok = (row_codes >= 0) & (col_codes >= 0)

        wide = np.full((len(rows), len(cols)), np.nan)
        wide[row_codes[ok], col_codes[ok]] = df[values].to_numpy()[ok]

        return pd.DataFrame(wide, index=pd.Index(rows, name=index), columns=pd.Index(cols, name=columns))

    def build_data(self
                   , cache_path:str
                   , data:str
                   , start:str=None
                   , end:str=None
                   , data_freq:str='W-FRI'
                   ):
        '''
        build data for weight calculation. Intermediate data is saved as parquet in cache_path and reused only while
        it still matches the current stock files and arguments (see __cache_data__)
        :param start:
        :param end:
        :return: True if success
        '''
        os.makedirs(cache_path, exist_ok=True)

        start_m, end = self.__proc_na_date(start, end, 53)

        if data == 'momentum':
            (flib.momentum(
                self.__cache_data__(cache_path, 'returns', start_m, end, data_freq)
                , lt=52, st=4)[start:end]
             .to_parquet(f'{cache_path}/momentum.parquet'))
        elif data == 'reversal':
            (flib.rev(
                self.__cache_data__(cache_path, 'returns', start_m, end, data_freq)
                , term=4)[start:end]
             .to_parquet(f'{cache_path}/reversal.parquet'))
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
            # firm size (company-wide market cap), so every share line of a company lands in the same size bucket
            (flib.size(
                self.__cache_data__(cache_path, 'cmktcap', start_m, end, data_freq)
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
            start = FULL_START
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
        path = f'{cache_path}/{data}.parquet'
        # reuse the file only if it was built from the current stock files with the same arguments
        sig = json.dumps({'version': CACHE_VERSION, 'source': self._source_signature(),
                          'start': str(start), 'end': str(end), 'freq': str(data_freq)}, sort_keys=True)

        meta = self._read_cache_meta(path)
        if meta is None or meta.get(CACHE_SIG_KEY, b'').decode() != sig:
            on_grid = None
            if data == 'returns':
                df = self.data(data, start=start, end=end).resample(data_freq).ffill() / 100
            elif data in ['dividend', 'dps', 'dpsnorm', 'dy', 'zdps']:
                df = self.data(data, start=start, end=end).resample(data_freq).max().fillna(0)
            else:
                raw = self.data(data, start=start, end=end)
                df = raw.ffill().resample(data_freq).ffill()
                grid = pd.date_range(raw.index.min(), raw.index.max(), freq=data_freq) if len(raw) else pd.DatetimeIndex([])
                on_grid = bool(len(raw)) and bool(raw.index.isin(grid).all())
            meta = self._write_cache(df, path, sig, on_grid)

        out = pd.read_parquet(path)
        if data == 'zprice' and start == FULL_START and meta.get(CACHE_ON_GRID_KEY) == b'1':
            self._zprice_weekly = (end, out)    # __zDPS__ reuses it instead of recomputing zprice
        return out

    def _source_signature(self):
        """
        Hash of (instrument, key, mtime, size) of every per-stock file of the loaded stocks.
        Changes whenever one of their stock files is added, re-downloaded or removed.
        """
        if self._source_sig is None:
            raw_keys = {k for k, v in self.data_master.items() if v.get('cls') != 'tag'}
            self._source_sig = self.files.signature(self.stock, raw_keys)
        return self._source_sig

    @staticmethod
    def _read_cache_meta(path):
        try:
            return pq.read_schema(path).metadata or {}
        except (OSError, pa.ArrowException):     # missing, unreadable or half-written -> rebuild
            return None

    @staticmethod
    def _read_cache_sig(path):
        meta = DataProcessor._read_cache_meta(path)
        return None if meta is None else meta.get(CACHE_SIG_KEY, b'').decode()

    @staticmethod
    def _write_cache(df, path, sig, on_grid=None):
        """same file df.to_parquet(path) writes, plus the signature (and on-grid flag) in the schema metadata"""
        table = pa.Table.from_pandas(df)
        meta = {**(table.schema.metadata or {}), CACHE_SIG_KEY: sig.encode()}
        if on_grid is not None:
            meta[CACHE_ON_GRID_KEY] = b'1' if on_grid else b'0'
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, path)
        return meta

    def _full_zprice(self, end):
        """full-history zprice, forward-filled; reuses the weekly file build_data just loaded when it is equivalent"""
        memo = self._zprice_weekly
        if memo is not None and memo[0] == end:
            return memo[1]
        return self.data('zprice', end=end).ffill()

    def __split__(self, start, end):
        fyshares = self.data('fyshares', start, end)

        # divisor: fyshares of the first instrument (sorted) reporting on the latest date
        latest = fyshares.iloc[-1].dropna()

        return fyshares / (latest.iloc[0] if len(latest) else float('nan'))

    # Derived frames below mix raw frames whose date index is the union over all loaded stocks. A stock's own value
    # is forward-filled on that union index BEFORE it is reindexed onto another frame's dates: reindex(method='ffill')
    # takes the previous row even when this stock has no value there. Nothing is back-filled from later prices
    # (the old closeprice .bfill() in __mktpct__ pulled prices from years ahead into B/P, E/P and the coupons).

    def __amktcap__(self, start, end):
        # free-float market cap of this listing: its own shares x its own price x its own free float (weights,
        # groupER mktval). mktcap (MVC) is company-wide and would give every share line of a company its full value
        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        price = self.data('closeprice', start, end).ffill().reindex(shares.index, method='ffill')
        cap = shares.multiply(price) / 1E6      # same million units as mktcap
        return cap.multiply(self.__ffpct(cap.index, start, end) / 100)

    def __cmktcap__(self, start, end):
        # company-wide market cap (MVC, all share classes) x this listing's free float: firm size for the SMB ranking
        mktcap = self.data('mktcap', start, end).ffill()
        return mktcap.multiply(self.__ffpct(mktcap.index, start, end) / 100)

    def __ffpct(self, index, start, end):
        # the leading gap before a stock's first free-float report is back-filled (slow-moving level, no price info)
        return self.data('freefloatpct', start, end).ffill().bfill().reindex(index, method='ffill')

    def __mktpct__(self, start, end):
        # share of the company's market value (MVC) in this listing
        shares = (self.data('fyshares', start, end) * 1E3).ffill()
        mktcap = (self.data('mktcap', start, end) * 1E6).ffill()
        closeprice = self.data('closeprice', start, end).ffill().reindex(shares.index, method='ffill')
        issuecap = shares.multiply(closeprice)

        return issuecap.divide(mktcap.reindex(issuecap.index, method='ffill'))

    def __aprice__(self, start, end):
        closeprice = self.data('closeprice', start, end).ffill()
        split = self.data('split', start, end).ffill().reindex(closeprice.index, method='ffill')

        return closeprice.multiply(split)

    def __zprice__(self, start, end):
        rtn = self.data('returns', start, end)
        zprice = (1 + rtn/100).cumprod()

        return zprice

    def __per_share(self, value, start, end, norm):
        '''company-level value -> per share of this listing, allocated by its share of the company's market value'''
        if norm:
            value = value.multiply(self.data('split', start, end).ffill().reindex(value.index, method='ffill'))
        value = value.multiply(self.data('mktpct', start, end).ffill().reindex(value.index, method='ffill'))
        shares = (self.data('fyshares', start, end) * 1E3).ffill().reindex(value.index, method='ffill')

        return value.divide(shares)

    def __EPS__(self, start, end):
        return self.__per_share(self.data('earnings', start, end, rolling=4).ffill(), start, end, False)

    def __EPSnorm__(self, start, end):
        return self.__per_share(self.data('earnings', start, end, rolling=4).ffill(), start, end, True)

    def __BPS__(self, start, end):
        return self.__per_share(self.data('bookvalue', start, end).ffill(), start, end, False)

    def __BPSnorm__(self, start, end):
        return self.__per_share(self.data('bookvalue', start, end).ffill(), start, end, True)

    def __DPS__(self, start, end):
        # dividends are a flow: never forward-filled themselves
        return self.__per_share(self.data('dividend', start, end), start, end, False)

    def __DPSnorm__(self, start, end):
        return self.__per_share(self.data('dividend', start, end), start, end, True)

    def __zDPS__(self, start, end):
        dps = self.data('dps', start, end)

        price = self.data('closeprice', start, end).ffill()
        price = price.reindex(dps.index, method='ffill')

        # dividend in zprice units (coupons for the backtest). zprice always comes from the full history, so a
        # zdps built for a recent window has the same scale as the full zprice the backtest runs on
        zprice = self._full_zprice(end)
        zprice = zprice.reindex(dps.index, method='ffill')

        return dps.divide(price).multiply(zprice)

    def __EP__(self, start, end):
        aprice = self.data('aprice', start, end).ffill()
        EPS = self.data('EPSnorm', start, end).ffill().reindex(aprice.index, method='ffill')

        return EPS.divide(aprice)

    def __BP__(self, start, end):
        aprice = self.data('aprice', start, end).ffill()
        BPS = self.data('BPSnorm', start, end).ffill().reindex(aprice.index, method='ffill')

        return BPS.divide(aprice)

    def __ROE__(self, start, end):
        BPS = self.data('BPS', start, end).ffill()
        EPS = self.data('EPS', start, end).ffill().reindex(BPS.index, method='ffill')

        return EPS.divide(BPS)

if __name__ == '__main__':

    dp = DataProcessor()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'{BaseData.dbpath}/members/master_{index}.parquet', engine='pyarrow')
    curr_index = index_master.iloc[-1:].dropna(axis=1)
    dp.load_stocks(curr_index.columns)

    shutil.rmtree('./cache')
    dp.build_data('./cache', 'momentum', '2020-01-01', '2025-01-01')
    dp.build_data('./cache', 'aprice', '2020-01-01', '2025-01-01')

    a=pd.date_range(start='2000-01-01', end='2025-02-28', freq='w-fri')
    # set a as monthly
    pd.Series(a, index=a).resample('M').nearest()
