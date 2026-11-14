import json
import os
from glob import glob
import pandas as pd
import datetime as dt
from tqdm import tqdm
from time import sleep

from src.data_manager import MasterDataManager
from src.rawstore import RawStore


class RawDataWriter:
    '''writes the downloaded bulk data into per-stock files (reading side: src/loadData.DataProcessor)'''
    def __init__(self, db_path, emas_name, dmas_name, raw_store=True):
        self.dmas_path = db_path + '/' + dmas_name
        self.emas_path = db_path + '/' + emas_name
        self.data_master = json.load(open(self.dmas_path, 'r'))
        self.db_path = db_path
        self.load_equity_master()
        # consolidated copy of the per-stock files that loadData reads (see src/rawstore.py); the per-stock files stay the source
        self.raw_store = RawStore(f'{db_path}/derived', f'{db_path}/db') if raw_store else None

    def load_equity_master(self):
        if os.path.exists(self.emas_path):
            self.equity_master = MasterDataManager(self.emas_path)
        else:
            raise FileNotFoundError('equity_master.db file not found')

    def save_tmp_row(self, col, data, start_date, end_date, frq, force_update=False):
        # if date is same, then no data. Set stat as False
        if start_date == end_date:
            stat = False
        else:
            stat = True

        col_data = self.equity_master.get_sub_value(col, data)
        if col_data is None or col_data == {} or force_update:
            col_data = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'frq': frq,
                'stat': stat
            }
            self.equity_master.update_sub_value(col, data, col_data)

            data_state = 0
        else:
            if end_date > pd.to_datetime(col_data['end']):
                self.equity_master.update_sub_value(col, f'{data}.end', end_date.strftime('%Y-%m-%d'))
                data_state = 1
            else:
                data_state = 2

        return data_state

    @staticmethod
    def dsws_column_reader(data, tmp):
        try:
            tmp = tmp.astype(float)
            tmp.name = data
        except:
            tmp = pd.Series(name=data)
        return tmp

    @staticmethod
    def rd_column_reader(data, tmp, cls):
        try:
            if cls == 'prc':
                tmp.columns = ['calcdate', data]
            else:
                tmp.columns = ['calcdate', 'fydate', data]
            tmp = tmp.reset_index().set_index(['calcdate']).drop(columns='Instrument')
            tmp.index = pd.to_datetime(tmp.index)
        except:
            tmp = pd.Series(name=data)
        return tmp

    def process(self, data, force_update=False):
        data_bulks = glob(f'{self.db_path}/db/bulk/*{data}*.csv')
        written = {}    # instrument -> (frame written to its stock file, [mtime_ns, size]); mirrored into the raw store afterwards

        for data_path in tqdm(data_bulks, desc=f'Processing {data}'):
            data_master_src = self.data_master[data]['src']
            data_master_cls = self.data_master[data]['cls']
            if data_master_src == 'dsws':
                data_bulk = pd.read_csv(data_path, index_col=0, header=0).iloc[3:]
                data_bulk.index = pd.to_datetime(data_bulk.index)
                data_bulk.columns = data_bulk.columns.str.strip()
                equ_list = data_bulk.columns.unique()
            else:
                data_bulk = pd.read_csv(data_path, index_col=0)
                if 'Instrument' not in data_bulk.columns:
                    data_bulk = data_bulk.T
                else:
                    data_bulk = data_bulk.set_index('Instrument')

                data_bulk.index = data_bulk.index.str.strip()
                equ_list = data_bulk.index.unique()

            for col in equ_list:
                if data_master_src == 'dsws':
                    tmprow = self.dsws_column_reader(data, data_bulk[col])
                else:
                    tmprow = self.rd_column_reader(data, data_bulk.loc[col], data_master_cls)

                start_date = tmprow.first_valid_index() or dt.datetime(1990, 1, 1)
                end_date = tmprow.last_valid_index() or dt.datetime(1990, 1, 1)
                frq = self.data_master[data]['frq']

                data_state = self.save_tmp_row(col, data, start_date, end_date, frq, force_update=force_update)

                if data_state == 1:
                    prev_data = pd.read_parquet(f'{self.db_path}/db/{col}_{data}.parquet', engine="pyarrow")
                    prev_data = prev_data.rename(columns={col:data})
                    tmprow = prev_data.combine_first(pd.DataFrame(tmprow))
                elif data_state == 2:
                    continue
                else:
                    tmprow = pd.DataFrame(tmprow)
                tmprow.index.name = 'date'
                tmprow['instrument'] = col
                tmprow.to_parquet(f'{self.db_path}/db/{col}_{data}.parquet', engine="pyarrow", compression="snappy")

                if self.raw_store is not None:
                    try:
                        st = os.stat(f'{self.db_path}/db/{col}_{data}.parquet')
                        written[col] = (tmprow, [st.st_mtime_ns, st.st_size])
                    except OSError:
                        pass

            # after save all data, delete data bulk
            while os.path.exists(data_path):
                try:
                    os.remove(data_path)
                except:
                    sleep(1)

        if written:
            # never let the store break ingestion: the stock files are the source and the next read repairs the store
            try:
                self.raw_store.upsert(data.lower(), written)
            except Exception as e:
                print(f'[RawStore] {data}: not updated ({type(e).__name__}: {e}); it is repaired on the next read')

        self.equity_master.commit(force=True)
        return True


if __name__ == '__main__':
    datalist = ['earnings', 'closeprice', 'return', 'fyshares', 'dividend', 'bookvalue', 'totalassets']
    processor = RawDataWriter('./data_master.json', './equity_master.json', './db/bulk')
    processor.process(datalist)