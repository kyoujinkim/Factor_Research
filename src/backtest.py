import shutil
import warnings
from glob import glob

import pandas as pd
import os
import bt
import src.loadData as ld

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_bt(w, p, d, stname:str=None, date_range:str=None):
    weight = w
    if date_range is not None:
        weight = weight[(weight.index >= date_range)]

    weight_info = [stname, weight.index[0], p.index[-1]]

    price_bt = p[(p.index >= weight_info[1]) * (p.index <= weight_info[2])][weight.columns]
    div_bt = d[(d.index >= weight_info[1]) * (d.index <= weight_info[2])][weight.columns]

    div_bt = div_bt.fillna(0)
    price_bt = price_bt.ffill()

    securities = [bt.CouponPayingSecurity(name, fixed_income=False) for name in weight.columns.values]
    s = bt.Strategy(weight_info[0]
                    , [bt.algos.WeighTarget(weight)
                    , bt.algos.Rebalance()
                    ]
                    , children=securities
                    )

    t = bt.Backtest(s, price_bt
                    , integer_positions=False
                    #, initial_capital=100000000000# 매우 큰값으로 설정. 결과값에는 영향이 없으나, 단주/ 커미션 부족 등의 에러 방지
                    #, commissions=lambda q, p: abs(q) * p * 0.00215
                    , additional_data={'coupons': div_bt}
                    , progress_bar=True
                    ,
                    )

    return t

if __name__ == "__main__":

    dp = ld.DataProcessor()
    dp.turnon_cache()

    # 1. index 선택
    index = 'RAY'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    dp.load_stocks(index_master.columns)

    bpath = './cache/us'
    #shutil.rmtree(bpath, ignore_errors=True)
    #os.makedirs(bpath, exist_ok=True)

    data_list = ['return', 'closeprice', 'dps']

    # build data
    for data in tqdm(data_list):
        dp.build_data(bpath, data, None, None)

    p = pd.read_parquet(f'{bpath}/closeprice.parquet', engine='pyarrow')
    d = pd.read_parquet(f'{bpath}/dps.parquet', engine='pyarrow')
    r = pd.read_parquet(f'{bpath}/return.parquet', engine='pyarrow')

    p_r = (1+r).cumprod()
    dy = (d / p * p_r)

    wsp = glob(f'./*.csv')
    bt_list = []
    for wp in wsp:
        w = pd.read_csv(wp, index_col=0, parse_dates=True)
        bt_list.append(run_bt(w, p_r, dy, wp.split('\\')[-1].split('.')[0]))

    res = bt.run(*bt_list)
    res.prices.to_csv(f'bt_plot.csv')

    res.display()