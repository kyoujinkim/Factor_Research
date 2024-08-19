import os
from glob import glob

import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt

def adjust_reverse(eachc, reverse):
    if reverse:
        adj_eachc = -eachc
        adj_eachc += (- adj_eachc.mean() - adj_eachc.mean())
    else:
        adj_eachc = eachc

    return pd.DataFrame(adj_eachc.values, columns=['ER'], index=adj_eachc.index)

def groupER(snapdate, country, model, avg_window, lr_window):

    ER_path = f'./output/{country}/Expected_Return/{model}_{avg_window}m_{lr_window}m_snapshot{snapdate}.csv'
    Beta_path = f'./output/{country}/Expected_Return/{model}_{avg_window}m_{lr_window}m_snapshot_betas_{snapdate}.csv'

    ER = pd.read_csv(ER_path, index_col=0, dtype=float, parse_dates=True)
    ER.columns = ['ER']
    ER = ER.sort_values('ER', ascending=False)[5:-5]
    mktval = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True).iloc[-1]
    industry = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True).iloc[-1]
    price = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True).iloc[-53:]
    price = price[ER.index]
    vol = (price.pct_change().iloc[1:].std() * np.sqrt(4))
    ER['Vol'] = vol
    ER['Sharpe'] = ER.ER / ER.Vol

    mktval_percentile = mktval.rank(pct=True)

    ER['mktval'] = mktval
    ER['mktvalER'] = ER['ER'] * ER['mktval']
    if country == 'korea':
        sector_length = 5
    else:
        sector_length = 8
    ER['industry'] = industry.str[:sector_length]

    Beta = pd.read_csv(Beta_path, index_col=0, dtype=float, parse_dates=True)
    Beta = Beta.loc[ER.index]
    Beta['mktval'] = mktval
    Beta['industry'] = industry.str[:sector_length]

    #calculate ER weighted by market value
    weighted_ER = ER['mktvalER'].sum() / ER['mktval'].sum()
    mean_ER = ER['ER'].mean()
    #calculate mktval weighted ER by size group (quantile)
    ER['size'] = pd.qcut(ER['mktval'], 3, labels=['small', 'mid', 'large'])
    size_ER = ER.groupby('size')['mktvalER'].sum() / ER.groupby('size')['mktval'].sum()
    size_mean_ER = ER.groupby('size')['ER'].mean()
    #calculate mktval weighted ER by industry group
    industry_ER = ER.groupby('industry')['mktvalER'].sum() / ER.groupby('industry')['mktval'].sum()
    industry_mean_ER = ER.groupby('industry')['ER'].mean()
    industry_beta = Beta.groupby('industry').mean()

    #calculate factor return
    factor_result = []
    factor_weights = glob(f'./weight/{country}/factor_weight/*.csv')
    for weight in factor_weights:
        factor_name = weight.split('\\')[-1].split('.')[0]
        curr_weight = pd.read_csv(weight, index_col=0, parse_dates=True).iloc[-1].dropna()
        ER[factor_name] = curr_weight
        factor_result.append([factor_name, (ER[factor_name] * ER['ER']).sum()])
    factor_result = pd.DataFrame(factor_result, columns=['factor', 'factor_return'])

    #get MV optimized Weight
    weightpath = glob(f'./weight/{country}/mvo_weight/*.csv')[-1]
    weight = pd.read_csv(weightpath, index_col=0, parse_dates=True).T
    weight['mktval'] = mktval_percentile.copy()
    #weight = weight[weight.mktval >= 0.2]
    weight = weight.T.iloc[-2].sort_values(ascending=False).head(40)

    tempER = ER.copy()
    tempER['mktvalpct'] = mktval_percentile.copy()
    tempER = tempER[tempER.mktvalpct >= 0.05]

    ERtop10 = tempER.sort_values('Sharpe', ascending=False).head(20)
    ERbot10 = tempER.sort_values('Sharpe', ascending=True).head(20)

    result_dict = {
        'total_ER': ER
        , 'beta': Beta
        , 'wmean_ER': pd.DataFrame([weighted_ER, mean_ER], index=['wmean_ER', 'mean_ER'], columns=['Index Expected Return'])
        , 'size_ER': pd.concat([size_ER,size_mean_ER], axis=1)
        , 'industry_ER': pd.concat([industry_ER,industry_mean_ER,industry_beta], axis=1)
        , 'factor_result': factor_result
        , 'ERtop+bot10': pd.concat([ERtop10, ERbot10], axis=0)[['industry', 'mktval','size', 'ER', 'Vol', 'Sharpe']]
        , 'optimized20': weight
    }

    weight_path = f'./output/0.export/{country}'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    #save each result in to csv file
    for key, value in result_dict.items():
        value.to_csv(f'./output/0.export/{country}/{key}.csv')