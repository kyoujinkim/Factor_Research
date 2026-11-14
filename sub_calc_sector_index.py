import os

import bt
import pandas as pd
from tqdm import tqdm

from src.backtest import run_bt
from src.calcWeight_2v import WeightBuilder
from src.loadData import DataProcessor

if __name__ == '__main__':
    for country, index in [
        #['us', 'RAY'],
        ['japan', 'TPX500'],
        ['europe', 'SXXP'],
        ['korea', 'KRX300'],
        ['china', 'MXCN'],
    ]:

        if not os.path.exists(f'./output/{country}/bm'):
            os.makedirs(f'./output/{country}/bm')

        bpath = f'./cache/{country}'

        dp = DataProcessor()
        dp.turnon_cache()
        dp.set_raw_store()

        # 2. index별로 저장되어있는 index_master 불러오기
        index_master = pd.read_parquet(f'{DataProcessor.dbpath}/members/master_{index}.parquet', engine='pyarrow')
        dp.load_stocks(index_master.columns)

        industry = dp.tag('sector')

        for data in tqdm(['zprice', 'zdps']):
            dp.build_data(bpath, data, None, None)

        # load price
        p = pd.read_parquet(f'{bpath}/zprice.parquet', engine='pyarrow')
        # load div
        d = pd.read_parquet(f'{bpath}/zdps.parquet', engine='pyarrow')

        '''load weight'''
        sector_list = industry['sector'].str[:2].drop_duplicates().tolist()
        sector_ts = []
        for sector in tqdm(sector_list, desc=f'Calculating {country} sector index'):
            # skip if sector is nan
            if sector!=sector or '<' in sector:
                continue
            # filter index_master by sector
            sector_index_master = index_master.loc[:, industry[industry['sector'].str[:2] == sector].index]
            # align columns
            unique_column = list(set(sector_index_master.columns) & set(p.columns) & set(d.columns))
            # if no common columns, skip
            if len(unique_column) == 0:
                continue
            sector_index_master = sector_index_master[unique_column]
            sector_p = p[unique_column]
            sector_d = d[unique_column]
            # skip if sector is 'nan'
            if sector!=sector:
                continue
            # calc market cap weight
            wb = WeightBuilder(sector_index_master, rebal_freq='m', data_freq='W-FRI')
            w = wb.run(start=None
                          , end=None
                          , weighting_method='mktcap'
                          , base_path=bpath
                          , workers=os.cpu_count()-3
                          , active_multiproc=True)

            res = bt.run(run_bt(sector, w, sector_p, sector_d, thresh=0))
            res.prices.to_csv(f'./output/{country}/bm/BM{sector}.csv')
            sector_ts.append(res.prices)

        sector_ts = pd.concat(sector_ts, axis=1).ffill()
        sector_ts.to_csv(f'output/0.export/{country}/sector_ts.csv')