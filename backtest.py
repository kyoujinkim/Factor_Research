import warnings
from glob import glob

import pandas as pd
import numpy as np
import datetime as dt
import os
import bt
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm

import quantstats as qs

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_bt(weight_path, price_bd, div_bd, date_range:str=None):
    weight = pd.read_csv(weight_path, index_col=0, parse_dates=True)
    if date_range is not None:
        weight = weight[(weight.index >= date_range)]

    weight_info = [weight_path.split('\\')[-1].split('.c')[0], weight.index[0], price_bd.index[-1]]

    price_bt = price_bd[(price_bd.index >= weight_info[1]) * (price_bd.index <= weight_info[2])][weight.columns]
    div_bt = div_bd[(div_bd.index >= weight_info[1]) * (div_bd.index <= weight_info[2])][weight.columns]

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

def get_result_csv(result):
    result_text = result.to_csv() + '\nTurnover'
    for bt_el in result.backtest_list:
        turnover_df = bt_el.turnover
        result_text += f',{round(turnover_df[turnover_df != 0].mean() * 100, 2)}%'
    result_text += '\nHerfindahl Index'
    for bt_el in result.backtest_list:
        herfindahl_df = bt_el.herfindahl_index
        result_text += f',{round(herfindahl_df[herfindahl_df != 0].mean() * 100, 2)}'

    return result_text

def sector_backtest(country, price_bd, div_bd):
    '''load weight'''
    weight_list = glob(f'./weight/{country}/sector*.csv')

    for weight in weight_list:
        t = run_bt(weight, price_bd, div_bd)

        res = bt.run(t)
        # res.set_date_range(start='2012-01-01')

        sec_name = weight.split('\\')[-1].split('_')[0]

        res.prices.to_csv(f'./output/{country}/industry_return/{sec_name}.csv')

def normal_backtest(country, price_bd, div_bd, avgwindow, datawindow):
    '''load weight'''
    bm_list = glob(f'./weight/{country}/factor_weight/BM.csv')
    weight_list = glob(f'./weight/{country}/240429/*{avgwindow}m+{datawindow}m*5p5p*.csv')
    weight_list = bm_list + weight_list

    bt_list = []
    for weight in weight_list:
        bt_list.append(run_bt(weight, price_bd, div_bd))

    res = bt.run(*bt_list)
    res.set_date_range(end='2024-03-29')

    res_text = get_result_csv(res)
    with open(f'./output/{country}/result_MVO_withBM_{rebal_freq}_0507_5p.csv', 'w') as out:
        out.write(res_text)
    print('\n\n')
    res.display()

    res.prices.to_csv(f'./output/{country}/prices_MVO_withBM_{rebal_freq}_0507_5p.csv')

    res.set_date_range(start='2019-01-01', end='2024-03-29')

    res_text = get_result_csv(res)
    with open(f'./output/{country}/result_MVO_withBM_{rebal_freq}_0507_5p_clip.csv', 'w') as out:
        out.write(res_text)
    print('\n\n')
    res.display()

    res.prices.to_csv(f'./output/{country}/prices_MVO_withBM_{rebal_freq}_0507_5p_clip.csv')

    return res


def general_backtest(country, price_bd, div_bd):
    '''load weight'''
    bm_list = glob(f'./weight/{country}/factor_weight/BM.csv')
    weight_list = glob(f'./weight/{country}/EquWeight_byQuantile/*sharpe*.csv')
    weight_list = bm_list + weight_list

    bt_list = []
    for weight in weight_list:
        bt_list.append(run_bt(weight, price_bd, div_bd))

    res = bt.run(*bt_list)

    res_text = get_result_csv(res)
    with open(f'./output/{country}/result_MVO_withBM_EquQSharpe.csv', 'w') as out:
        out.write(res_text)
    print('\n\n')
    res.display()

    res.prices.to_csv(f'./output/{country}/prices_MVO_withBM_EquQSharpe.csv')

    return res


# C:\Users\NHWM\anaconda3\envs\Factor_Research\Lib\site-packages\ffn\core.py
# change TRADING_DAYS_PER_YEAR => 52 # default value was 252

if __name__ == "__main__":
    country_list = [
        ['us', 3, 12]
        , ['europe', 6, 12]
        , ['korea', 3, 60]
        , ['japan', 6, 60]
    ]
    #country_list = list(itertools.product(country_list, avgwindow_list, datawindow_list))

    for country, avgwindow, datawindow in country_list:

        if avgwindow >= datawindow:
            continue

        rebal_freq = 'm'

        '''load data'''
        price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0,
                               parse_dates=True)
        div_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_dividend.csv', dtype=float, index_col=0,
                             parse_dates=True)

        # consider Dividend Date, adjust dividend date
        div_bd = div_bd.shift(-1).fillna(0)

        # cai 불러오기open
        cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
        cai_bd = cai_bd.reindex(price_bd.index, method='bfill').ffill()
        cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
        cai_bd = cai_bd.fillna(1)

        # cai를 기준으로 share관련 지표들 수정
        price_bd *= cai_bd
        div_bd *= cai_bd

        #normal_backtest(country, price_bd, div_bd, avgwindow, datawindow)

        res = general_backtest(country, price_bd, div_bd)
        #sector_backtest(country, price_bd, div_bd)
        #for col in res.prices.columns:
        #    res.prices[col].to_csv(f'./output/{country}/industry_return/{col}.csv')

        #for col in res.prices.columns:
        #    res.prices[col].to_csv(f'./output/{country}/{col}.csv')
        #res.display()
        #plt.show()
        res = qs.reports.html(res.prices[res.prices.columns[-2]], res.prices[res.prices.columns[-1]]
                        , output=f'./output/{country}/FScoeff1.html')

    print('done')