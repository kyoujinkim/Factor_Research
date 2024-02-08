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

def run_bt(weight_path):
    weight = pd.read_csv(weight_path, index_col=0, parse_dates=True)
    weight_info = weight_path.split('\\')[-1].split('.')[0].split('_')

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

# C:\Users\NHWM\anaconda3\envs\Factor_Research\Lib\site-packages\ffn\core.py
# change TRADING_DAYS_PER_YEAR => 52 # default value was 252

country = 'europe' #'us' or 'korea' or 'japan' or 'europe'
rebal_freq = 'm'

'''load data'''
price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
div_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_dividend.csv', dtype=float, index_col=0, parse_dates=True)

#consider Dividend Date, adjust dividend date
div_bd = div_bd.shift(-1).fillna(0)

'''load weight'''
weight_list = glob(f'./weight/{country}/*.csv')

if __name__ == "__main__":


    bt_list = []
    for weight in weight_list:
        bt_list.append(run_bt(weight))

    res = bt.run(*bt_list)
    #res.set_date_range(start='2012-01-01')

    res.prices.to_csv(f'./output/{country}/price_bm_{rebal_freq}.csv')
    res_text = get_result_csv(res)
    with open(f'./output/{country}/result_bm_{rebal_freq}.csv', 'w') as out:
        out.write(res_text)
    print('\n\n')
    res.display()

    #for col in res.prices.columns:
    #    res.prices[col].to_csv(f'./output/{country}/{col}.csv')
    #res.display()
    #plt.show()
    #qs.reports.html(res.prices['FScoeff1+equal'], res.prices['bm+mktval']
    #                , output=f'./output/{country}/FScoeff1.html')

    print('done')