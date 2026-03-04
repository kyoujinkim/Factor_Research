import pickle
from typing import List

import pandas as pd
from tqdm import tqdm

from src.calcER_2v import Calculator
from src.loadData import DataProcessor

import itertools

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def calculate_Expected_Return(country, index, model, windows: List, end: str = None):
    '''
    기대수익률을 계산

    :param country: 국가명
    :param model: 모델명 abbreviation (e.g. CAPM, ff5 ...)
    :param avg_window: 미래 팩터 수익률로 사용하기 위한 평균값 window
    :param window: 회귀계수를 계산하기 위한 데이터 window
    :param end: 기대수익률을 계산할 최종날짜
    :return:
    '''
    avg_window, window = windows

    basewindow = window+2  # 넉넉하게 부여
    print(f'country: {country}')

    dp = DataProcessor()
    dp.turnon_cache()

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    dp.load_stocks(index_master.columns)

    for data in tqdm(['zprice']):
        dp.build_data(f'./cache/{country}', data, None, None)

    # load price
    p = pd.read_parquet(f'./cache/{country}/zprice.parquet', engine='pyarrow')
    rf = pd.read_excel(f'D:/Factor_DB/rf.xlsx', sheet_name='rf', index_col=0, parse_dates=True)[[country]].iloc[1:].astype(float)
    rf.index = pd.to_datetime(rf.index)

    # load factor data
    bm = pd.read_csv(f'./output/{country}/bm/BM.csv', dtype=float, index_col=0, parse_dates=True)
    # load factor data
    factors = {'bm':'BM',
               'hml':'HML',
               'smb':'SMB',
               'mom':'MOM',
               'op':'OP',
               'inv':'INV',
               'rev':'REV',
               'hs':'HS',
               }
    factor_dict = {}
    for f in factors:
        factor_dict[f] = pd.read_csv(f'./output/{country}/bm/{factors[f]}.csv', dtype=float, index_col=0, parse_dates=True)

    cal = Calculator(index=index_master, price=p,  rf=rf, bm=bm)

    c_ts = cal.run(method=model, resample='M', basewindow=basewindow, window=window, factors=factor_dict, avg_window=avg_window)
    save_pickle(c_ts, f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m.pkl')
    c_filter = [c for c in c_ts if c is not None]
    res = cal.evaluate(c_filter, window=1).dropna()
    res_profile = pd.concat([res.mean(), res.median()], axis=1)
    res_data = pd.concat([res.mean(), res.median()], axis=0)
    res_data.name = f'{model}_{avg_window}m_{window}m'
    print(res_profile)

    return res_data

if __name__ == '__main__':
    country = 'us'
    index = 'RAY'
    model_list = ['hs']# capm, 'ff3', 'ff5', 'ch4', 'sr2', 'rv4']
    window_list = [[3, 12], [6, 12], [3, 36], [6, 36], [12, 36], [3, 60], [6, 60], [12, 60]]
    param = itertools.product(model_list, window_list)
    total = []
    for p in param:
        c = calculate_Expected_Return(country, 'RAY', *p)
        total.append(c)

    pd.concat(total, axis=1).to_csv(f'result_{country}_rv.csv')