import json
import os
from glob import glob
import datetime as dt
from time import sleep

import numpy as np
import pandas as pd
from tqdm import tqdm
import refinitiv.data as rd
import DatastreamPy as dsws
from configparser import ConfigParser
from src.data_manager import MasterDataManager


class DataDownloader:
    def __init__(self, app_key, id, pw, db_path, emas_name, dmas_name):
        self.app_key = app_key
        self.id = id
        self.pw = pw
        session = rd.session.desktop.Definition(app_key=self.app_key).get_session()
        rd.session.set_default(session)
        session.open()
        self.ds_session = dsws.DataClient(username=self.id, password=self.pw)
        self.db_path = db_path
        self.emas_path = db_path + '/' + emas_name
        self.dmas_path = db_path + '/' + dmas_name

        # read equity master(dict file)
        self.load_equity_master()

        # read data master(dict file)
        self.data_master = {}
        self.load_data_master()

        self.index = None
        self.index_master = None

    def load_equity_master(self):
        if os.path.exists(self.emas_path):
            self.equity_master = MasterDataManager(self.emas_path)
        else:
            raise FileNotFoundError('equity_master.db file not found')

    def load_data_master(self):
        if os.path.exists(self.dmas_path):
            self.data_master = json.load(open(self.dmas_path, 'r'))
        else:
            raise FileNotFoundError('data_master.json file not found')

    def assign_index(self, index, exist=False):
        '''
        Assign index name to dataDownloader object
        e.g. 'RAY', 'SXXP', 'TPX500'
        :param index: from bloomberg index ticker
        :param exist: if True, load existing index master from parquet file
        :return: None, internal self's index_master is updated
        '''
        self.index = index
        if exist and os.path.exists(f'{self.db_path}/members/master_{index}.parquet'):
            self.index_master = pd.read_parquet(f'{self.db_path}/members/master_{index}.parquet', engine='pyarrow')
        else:
            self.index_master = self.build_index_master(index)

    def get_data(self, *args, **kwargs):
        try:
            return rd.get_data(*args, **kwargs)
        except Exception as e:
            print(e)
            err_code = e.args[0]
            if 400 <= err_code < 500:
                raise f"Client Error: {e.args[1]}"
            elif 500 <= err_code < 600:
                raise f"Server Error: {e.args[1]}"
            else:
                return -1

    def get_data_ds(self, *args, **kwargs):
        try:
            return self.ds_session.get_data(*args, **kwargs)
        except Exception as e:
            print(e)
            return -1

    @staticmethod
    def read_member_xl(file_path):
        info = file_path.split('\\')[-1].split('.')[0][-12:-1]
        port_date = dt.datetime.strptime(info, '%b %d %Y')
        # read xlsx
        memb_xl = pd.read_excel(file_path, engine='openpyxl')
        isin_list = memb_xl.filter(regex=('ISIN*')).iloc[:, 0].dropna().drop_duplicates()

        return isin_list, port_date

    def build_index_master(self, index):
        list_memb = glob(f'{self.db_path}/members/{index}/*.xlsx')

        total_df = []

        for memb in tqdm(list_memb):
            isin_list, port_date = self.read_member_xl(memb)
            total_df.append(pd.Series(1, index=isin_list, name=port_date))

        index_master = pd.concat(total_df, axis=1).T.sort_index(ascending=True)

        index_master.to_parquet(f'{self.db_path}/members/master_{index}.parquet', engine='pyarrow', compression='snappy')

        return index_master

    def process_tag(self, tag, force_update=False):
        '''
        download tag data(name, sector, parent, etc.) and save it to equity master
        :param tag: 'issuename', 'sector', 'name', 'country', 'currency'
        :param force_update: if True, force update data
        :return: None
        '''
        # set initial data parameters`
        current_date = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, dt.datetime.today().day)
        current_date -= dt.timedelta(days=1)

        # get unique equities
        list_equ = []
        for equ in self.index_master.columns:
            equ_tag = self.equity_master.get_sub_value(equ, tag)
            # if data is not in equity master, download data
            if equ_tag is None or equ_tag == '' or force_update:
                list_equ.append(equ)

        if len(list_equ) == 0:
            return False

        # download data
        data_total = []
        if self.data_master[tag]['cls'] == 'tag':
            if self.data_master[tag]['src'] == 'dsws':
                for i in tqdm(range(0, len(list_equ), 50), desc=f'Downloading {tag}'):
                    try:
                        data = self.get_data_ds(
                            tickers=', '.join(list_equ[i:i + 50]),
                            fields=[self.data_master[tag]['field']],
                            kind=0,  # 0 for tag
                        ).set_index('Instrument')[['Value']]
                    except:
                        continue
                    data.index = data.index.str.strip()
                    data.columns = [tag]
                    data_total.append(data)
            else:
                for i in tqdm(range(0, len(list_equ), 1000), desc=f'Downloading {tag}'):
                    try:
                        data = self.get_data(
                            universe=list_equ[i:i + 1000],
                            fields=[self.data_master[tag]['field']],
                        ).set_index('Instrument')
                    except:
                        continue
                    data.columns = [tag]
                    data_total.append(data)

        if len(data_total) > 0:
            data_total_df = pd.concat(data_total, axis=0)

            for equ in list_equ:
                if equ not in data_total_df.index:
                    continue
                self.equity_master.update_sub_value(equ, tag, str(data_total_df.loc[equ].values[0]))

            self.equity_master.commit(force=True)
            return True
        else:
            return False

    def process(self, prev_months, post_months, data, currency='USD', currency_ds='U$', force_update=False):
        '''
        Run bulk data download
        :param prev_months: previous month to download for margin
        :param post_months: post month to download for margin
        :param data: which data to download
        :param currency: currency to convert data, e.g. 'USD', 'EUR', 'JPY'
        :param currency_ds: currency for Datastream, e.g. 'U$', 'EU', 'Y'
        :return: Bool, if success, return True
        '''
        # set initial data parameters
        current_date = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, dt.datetime.today().day)
        current_date -= dt.timedelta(days=1)

        # build index members' database
        print(f'Building {self.index} index master database...')
        index_curr = self.index_master.iloc[-1].dropna().index.tolist()

        tmp_list = []
        for equ in tqdm(self.index_master.columns):
            equ_start_date = self.index_master[equ].first_valid_index() - pd.DateOffset(months=prev_months)

            equ_end_date = self.index_master[equ].last_valid_index()
            # set end date as current date if end date is later than current date
            equ_end_date = min(equ_end_date + pd.DateOffset(months=post_months), current_date)
            equ_data = self.equity_master.get_sub_value(equ, data)

            # if data is not in equity master, download data
            if equ_data is None or force_update:
                tmp_list.append([equ, equ_start_date, equ_end_date])
            # if data is not up-to-date even though it is currently in index, download data
            elif dt.datetime.strptime(equ_data['end'], "%Y-%m-%d") < equ_end_date and equ in index_curr:
                equ_start_date = dt.datetime.strptime(equ_data['end'], "%Y-%m-%d") - pd.DateOffset(days=3)
                tmp_list.append([equ, equ_start_date, equ_end_date])
            else:
                continue

        tmp_df = pd.DataFrame(tmp_list, columns=['equ', 'start', 'end'])
        tmp_df = tmp_df.sort_values(by=['end', 'start'])
        tmp_df['labels'] = tmp_df.start.dt.strftime('%Y-%m-%d') + '_' + tmp_df.end.dt.strftime('%Y-%m-%d')
        #tmp_df['labels'] = tmp_df['end'].dt.strftime('%Y-%m-%d')
        labels = tmp_df.labels.unique()

        with tqdm(labels) as pbar:
            for label in tqdm(labels):
                # if file exists, skip
                tmp_df2 = tmp_df[tmp_df.labels == label]
                # if data is more than 100, download data seperately by 100 equities
                step_size = 50
                for i in range(0, len(tmp_df2), step_size):
                    if os.path.exists(f'{self.db_path}/db/bulk/{data}_{label}_{i}.csv'):
                        continue
                    tmp_df2_sub = tmp_df2.iloc[i:i + step_size].copy()
                    pbar.set_description(f'Downloading data: {data} - {label} - {i} to {i + 50} / {tmp_df2.equ.count()} equities')

                    # set download date, friday of the week
                    tmp_df_start_date = tmp_df2_sub.start.min()
                    tmp_df_start_date = tmp_df_start_date - pd.DateOffset(days=(tmp_df_start_date.weekday() +3) % 7)
                    tmp_df_end_date = tmp_df2_sub.end.max()
                    tmp_df_end_date = tmp_df_end_date - pd.DateOffset(days=(tmp_df_end_date.weekday() + 3) % 7)

                    if self.data_master[data]['src'] == 'dsws':
                        if data in ['returns', 'fyshares']:
                            fieldname = self.data_master[data]['field']
                        else:
                            fieldname = f"X({self.data_master[data]['field']})~{currency_ds}"
                        rd_data = self.get_data_ds(
                            tickers=', '.join(tmp_df2_sub.equ.to_list()),
                            fields=[fieldname],
                            start=tmp_df_start_date.strftime('%Y-%m-%d'),
                            end=tmp_df_end_date.strftime('%Y-%m-%d'),
                            freq=self.data_master[data]['frq']
                        )
                    else:
                        rhs = self.data_master[data]['rh'].split(';')
                        fields = [f"{self.data_master[data]['field']}.{rh}" for rh in rhs]
                        fields += [self.data_master[data]['field']]
                        # download data
                        rd_data = self.get_data(
                            universe=tmp_df2_sub.equ.to_list(),
                            fields=fields,
                            parameters={
                                'SDate': tmp_df_start_date.strftime('%Y-%m-%d'),
                                'EDate': tmp_df_end_date.strftime('%Y-%m-%d'),
                                'Period': self.data_master[data]['period'],
                                'Frq': self.data_master[data]['frq'],
                                'Curn': currency,
                            }
                        )
                    if type(rd_data) != pd.DataFrame:
                        rd_data = pd.DataFrame(columns=tmp_df2_sub.equ.to_list())
                    rd_data.to_csv(f'{self.db_path}/db/bulk/{self.index}_{data}_{label}_{i}.csv', encoding='utf-8-sig')

                pbar.update(1)

        return True