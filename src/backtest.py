import hashlib
import json
import os
import shutil
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
    # coupons sit on the same W-FRI grid as prices; a week without a row paid nothing. (method='nearest' used to copy
    # the last dividend week onto every later week, and gave an all-NaN frame when the window started after it)
    d_bt = d[unique_code].reindex(p_bt.index).fillna(0)

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


'''
Incremental factor index.
A full run backtests every rebalance date since the start (minutes per factor). Rebalance dates before the first
new or changed weight row produce the same path as last time, so only the tail is backtested and spliced onto the
saved series at a common date (same result as a full run up to floating-point error, ~1e-14).
Assumes the price/coupon history before the restart date did not change: after repairing a stock's history,
run once with full=True.
'''
BT_STATE_VERSION = 1


def _state_path(out_csv):
    return f'{out_csv}.state.json'


def _weight_row_hashes(w, p_columns, thresh):
    '''one hash per rebalance row, taken from the rows exactly as run_bt passes them to bt'''
    w = w.dropna(thresh=thresh)
    w = w[sorted(set(w.columns) & set(p_columns))]
    return {dt.strftime('%Y-%m-%d'): hashlib.sha1(','.join(f'{c}:{v!r}' for c, v in row.dropna().items()).encode()).hexdigest()
            for dt, row in w.iterrows()}


def plan_factor_index(w, p_columns, out_csv, thresh:int=30, full:bool=False):
    '''
    Decide where the backtest of weights w has to start.
    :return: dict(restart=Timestamp or None for a full run, reason=str, hashes=..., prev=saved series or None)
    '''
    hashes = _weight_row_hashes(w, p_columns, thresh)
    plan = {'restart': None, 'hashes': hashes, 'prev': None, 'reason': ''}
    if full:
        plan['reason'] = 'full requested'
        return plan
    try:
        with open(_state_path(out_csv), 'r') as f:
            state = json.load(f)
        prev = pd.read_csv(out_csv, index_col=0, parse_dates=True)
    except (OSError, ValueError):
        plan['reason'] = 'no saved series/state'
        return plan
    if state.get('version') != BT_STATE_VERSION or state.get('thresh') != thresh:
        plan['reason'] = 'state from another version/thresh'
        return plan

    old = state.get('weights', {})
    changed = [dt for dt, h in hashes.items() if old.get(dt) != h] + [dt for dt in old if dt not in hashes]
    first = min(changed) if changed else max(hashes)
    # restart on the rebalance date before the first difference (a removed date needs the one before it). Restarting
    # earlier gives the same path; restarting on `first` itself can leave a one-day backtest when it is also the last
    # price date, where bt's extra leading day makes the coupon NaN and the run fails.
    restart = max((dt for dt in hashes if dt < first), default=None)
    if restart is None or restart <= min(hashes):
        plan['reason'] = f'first rebalance date changed ({first})'
        return plan
    plan.update(restart=pd.Timestamp(restart), prev=prev,
                reason=f'{len(changed)} new/changed rebalance date(s), restart {restart}')
    return plan


def run_factor_index(strat, w, p, d, out_csv, plan, thresh:int=30):
    '''backtest (fully, or from plan['restart'] and spliced), then save the series and its state'''
    series = None
    if plan['restart'] is not None:
        tail = bt.run(run_bt(strat, w, p, d, date_range=plan['restart'], thresh=thresh)).prices
        tail = tail[tail.index >= plan['restart']]          # drop the extra day bt puts before its first date
        prev = plan['prev']
        anchor = tail.index[0]
        if anchor in prev.index and list(prev.columns) == list(tail.columns):
            scale = prev.loc[anchor].iloc[0] / tail.iloc[0, 0]
            series = pd.concat([prev[prev.index < anchor], tail * scale])
        else:
            plan['reason'] += f' -> anchor {anchor.date()} not in saved series, full run instead'
    if series is None:
        series = bt.run(run_bt(strat, w, p, d, thresh=thresh)).prices

    # series first, then state: a stale state only makes the next run restart earlier, never splice wrongly
    series.to_csv(f'{out_csv}.tmp')
    os.replace(f'{out_csv}.tmp', out_csv)
    with open(f'{_state_path(out_csv)}.tmp', 'w') as f:
        json.dump({'version': BT_STATE_VERSION, 'thresh': thresh, 'weights': plan['hashes']}, f)
    os.replace(f'{_state_path(out_csv)}.tmp', _state_path(out_csv))
    return series


if __name__ == "__main__":

    dp = ld.DataProcessor()
    dp.turnon_cache()

    # 1. index 선택
    index = 'MXCN'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'{ld.BaseData.dbpath}/members/master_{index}.parquet', engine='pyarrow')
    dp.load_stocks(index_master.columns)

    bpath = './cache/china'
    shutil.rmtree(bpath, ignore_errors=True)
    os.makedirs(bpath, exist_ok=True)

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