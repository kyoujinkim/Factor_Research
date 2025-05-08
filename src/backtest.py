import warnings
from glob import glob

import pandas as pd
import bt
import src.loadData as ld

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_bt(strat, w, p, d, date_range:str=None, thresh:int=0):
    # drop if data is less than 50
    w = w.dropna(thresh=thresh)

    if date_range is not None:
        w = w[(w.index >= date_range)]

    w_info = [strat, w.index[0], p.index[-1]]

    unique_code = list(set(w.columns)&set(p.columns)&set(d.columns))

    w = w[unique_code]

    p_bt = p[(p.index >= w_info[1]) * (p.index <= w_info[2])][unique_code]
    d_bt = d[(d.index >= w_info[1]) * (d.index <= w_info[2])][unique_code].reindex(p_bt.index, method='nearest')

    securities = [bt.CouponPayingSecurity(name, fixed_income=False) for name in w.columns.values]
    s = bt.Strategy(w_info[0]
                    , [bt.algos.WeighTarget(w)
                    , bt.algos.Rebalance()
                    ]
                    , children=securities
                    )

    t = bt.Backtest(s, p_bt
                    , integer_positions=False
                    #, commissions=lambda q, p: abs(q) * p * 0.00215
                    , additional_data={'coupons': d_bt}
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
        bt_list.append(run_bt(wp, w, p_r, dy, wp.split('\\')[-1].split('.')[0]))

    res = bt.run(*bt_list)
    res.prices.to_csv(f'bt_plot.csv')

    res.display()