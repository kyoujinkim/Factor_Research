import argparse
import os
import pickle
import shutil
from glob import glob

import bt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from src.loadData import DataProcessor
from src.calcWeight_2v import WeightBuilder
from src.calcER_2v import Calculator
from src.backtest import run_bt

def calc_factor_weight(country, index, factorlist, recalc_w=False):

    if not os.path.exists(f'./weight/{country}/factor_weight'):
        os.makedirs(f'./weight/{country}/factor_weight')

    bpath = f'./cache/{country}'

    dp = DataProcessor()
    dp.turnon_cache()

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    dp.load_stocks(index_master.columns)

    for key in factorlist.keys():

        factor = factorlist[key]
        data_list = ['amktcap', 'returns']
        data_list.append(factor)

        if recalc_w:
            w_prev = pd.DataFrame()
            w_end = None
        else:
            w_prev = pd.read_csv(f'./weight/{country}/factor_weight/{key}.csv', index_col=0, parse_dates=True)
            w_end = (w_prev.index[-1]-relativedelta(weeks=5)).strftime('%Y-%m-%d')

        # build data
        for data in tqdm(data_list):
            dp.build_data(bpath, data, w_end, None)

        wb = WeightBuilder(index_master, rebal_freq='m', data_freq='W-FRI')
        if key == 'BM':
            w = wb.run(start=w_end
                          , end=None
                          , weighting_method='mktcap'
                          , weight_limit=0.3
                          , base_path=bpath
                          , workers=os.cpu_count()-3
                          , active_multiproc=True)
        else:
            w = wb.run(start=w_end
                       , end=None
                       , weighting_method='longshort'
                       , factor=factor
                       , factor_n=0.2
                       , base_path=bpath
                       , workers=os.cpu_count()-3
                       , active_multiproc=True)

        w = w_prev.combine_first(w)
        w.to_csv(f'./weight/{country}/factor_weight/{key}.csv')

    # initialize
    dp.turnoff_cache()
    dp.unload_stocks()

    return True

def calc_factor_index(country, index, factorlist):

    if not os.path.exists(f'./output/{country}/bm'):
        os.makedirs(f'./output/{country}/bm')

    bpath = f'./cache/{country}'

    dp = DataProcessor()
    dp.turnon_cache()

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    dp.load_stocks(index_master.columns)

    for data in tqdm(['zprice','zdps']):
        dp.build_data(bpath, data, None, None)

    # load price
    p = pd.read_parquet(f'{bpath}/zprice.parquet', engine='pyarrow')
    # load div
    d = pd.read_parquet(f'{bpath}/zdps.parquet', engine='pyarrow')

    '''load weight'''
    factor_ts = []
    for factor in factorlist:
        if factor == 'HS':
            res = pd.read_csv(f'D:/Factor_DB/external_factor/{country}/hs_delta.csv', index_col=0, parse_dates=True)[['HS_Delta']]
            res.columns = [factor]
        else:
            weight = pd.read_csv(f'./weight/{country}/factor_weight/{factor}.csv', index_col=0, parse_dates=True)
            res = bt.run(run_bt(factor, weight, p, d, thresh=30)).prices
        res.to_csv(f'./output/{country}/bm/{factor}.csv')
        factor_ts.append(res)

    factor_ts = pd.concat(factor_ts, axis=1).ffill()
    factor_ts.to_csv(f'output/0.export/{country}/factor_hs.csv')

    return True

def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def calculate_Expected_Return(country, index, model, avg_window, window, factors, end: str = None, calc_all:bool=False):
    '''
    기대수익률을 계산

    :param country: 국가명
    :param model: 모델명 abbreviation (e.g. CAPM, ff5 ...)
    :param avg_window: 미래 팩터 수익률로 사용하기 위한 평균값 window
    :param window: 회귀계수를 계산하기 위한 데이터 window
    :param end: 기대수익률을 계산할 최종날짜
    :return:
    '''

    if not os.path.exists(f'./output/{country}/Expected_Return'):
        os.makedirs(f'./output/{country}/Expected_Return')

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
    bm = pd.read_csv(f'./output/{country}/bm/BM.csv', dtype=float, index_col=0, parse_dates=True)

    # load factor data
    factor_dict = {}
    for f in factors:
        factor_dict[f] = pd.read_csv(f'./output/{country}/bm/{f.upper()}.csv', dtype=float, index_col=0, parse_dates=True)

    cal = Calculator(index=index_master, price=p,  rf=rf, bm=bm)

    c = cal.run_snapshot(method=model, resample='W', basewindow=basewindow, window=window, factors=factor_dict, avg_window=avg_window, end=end)
    betas = cal.run_snapshot(method='ff6', resample='M', basewindow=basewindow, window=window, factors=factor_dict, avg_window=avg_window, end=end, view_betas=True)
    if country=='korea':
        rbeta = cal.run_snapshot(method='sr2', resample='M', basewindow=basewindow, window=window, factors=factor_dict, avg_window=avg_window, end=end, view_betas=True)
        betas = pd.concat([betas, rbeta[['beta_sr']]], axis=1)

    c.to_csv(f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m_snapshot{c.name.strftime("%Y-%m-%d")}.csv')
    betas.to_csv(f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m_snapshot_betas_{c.name.strftime("%Y-%m-%d")}.csv')

    if calc_all:
        c_ts = cal.run(method=model, resample='M', basewindow=basewindow, window=window, factors=factor_dict, avg_window=avg_window)
        save_pickle(c_ts, f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m.pkl')

    return c.name.strftime("%Y-%m-%d")

def groupER(snapdate, country, model, avg_window, lr_window, mktpct=0.7, factors=None):
    '''
    make formatted output for Expected Return
    :param snapdate: date of snapshot
    :param country: country name(korea, us, japan, europe)
    :param model: model name(capm, sr2, ch4)
    :param avg_window: average window
    :param lr_window: long term window
    :param mktpct: market value percentile(larger than x)
    :return: None
    '''

    ER_path = f'./output/{country}/Expected_Return/{model}_{avg_window}m_{lr_window}m_snapshot{snapdate}.csv'
    Beta_path = f'./output/{country}/Expected_Return/{model}_{avg_window}m_{lr_window}m_snapshot_betas_{snapdate}.csv'

    ER = pd.read_csv(ER_path, index_col=0, dtype=float, parse_dates=True) * 100
    ER.columns = ['ER']
    ER = ER.sort_values('ER', ascending=False)#[5:-5]

    dp = DataProcessor()
    dp.turnon_cache()

    # 2. index별로 저장되어있는 index_master 불러오기
    dp.load_stocks(ER.index)

    mktval = dp.data('amktcap', end=snapdate).iloc[-1]
    vol = dp.data('returns', end=snapdate).iloc[-52:].std() * np.sqrt(4)
    industry = dp.tag('sector')
    industry = industry[industry.sector!='<NA>']
    compname = dp.tag('name')
    industryname = pd.read_csv(f'./output/industry_map_gics.csv', dtype=str).set_index('Code')

    ER['Name'] = compname.loc[ER.index]
    ER['Vol'] = vol
    ER['Sharpe'] = ER.ER / ER.Vol

    mktval_percentile = mktval.rank(pct=True)

    ER['mktval'] = mktval
    ER['mktvalER'] = ER['ER'] * ER['mktval']
    ER['sector'] = industry.sector.str[:2]
    ER['industry'] = industry.sector.str[:6]

    Beta = pd.read_csv(Beta_path, index_col=0, dtype=float, parse_dates=True)
    Beta = Beta.loc[ER.index]
    Beta['mktval'] = mktval
    Beta['sector'] = industry.sector.str[:2]
    Beta['industry'] = industry.sector.str[:6]
    # if country is korea, include reverse beta
    if country=='korea':
        betalist = ['intercept','beta_bm','beta_hml','beta_smb','beta_op','beta_inv','beta_mom','beta_sr']
    else:
        betalist = ['intercept','beta_bm','beta_hml','beta_smb','beta_op','beta_inv','beta_mom']

    #calculate ER weighted by market value
    weighted_ER = ER['mktvalER'].sum() / ER['mktval'].sum()
    mean_ER = ER['ER'].mean()
    #calculate mktval weighted ER by size group (quantile)
    ER['size'] = pd.qcut(ER['mktval'], 3, labels=['small', 'mid', 'large'])
    size_ER = ER.groupby('size')['mktvalER'].sum() / ER.groupby('size')['mktval'].sum()
    size_mean_ER = ER.groupby('size')['ER'].mean()
    #calculate mktval weighted ER by industry group
    industry_ER = ER.groupby('sector')['mktvalER'].sum() / ER.groupby('sector')['mktval'].sum()
    industry_ER.name = 'wER'
    industry_mean_ER = ER.groupby('sector')['ER'].mean()
    industry_beta = Beta.groupby('sector')[betalist].mean()
    industry_name = industryname.loc[industry_ER.index, 'Sector']

    subindustry_ER = ER.groupby('industry')['mktvalER'].sum() / ER.groupby('industry')['mktval'].sum()
    subindustry_ER.name = 'wER'
    subindustry_mean_ER = ER.groupby('industry')['ER'].mean()
    subindustry_beta = Beta.groupby('industry')[betalist].mean()
    subindustry_name = industryname.loc[subindustry_ER.index, 'Sector']

    #calculate factor return
    factor_result = []
    for factor in factors:
        fpath = f'./weight/{country}/factor_weight/{factor}.csv'
        curr_weight = pd.read_csv(fpath, index_col=0, parse_dates=True).iloc[-1].dropna()
        ER[factor] = curr_weight
        factor_result.append([factor, (ER[factor] * ER['ER']).sum()])
    factor_ER = pd.DataFrame(factor_result, columns=['factor', 'ER'])
    ER = pd.concat([ER, Beta[betalist]], axis=1)

    tempER = ER.copy()
    tempER['mktvalpct'] = mktval_percentile.copy()
    tempER = tempER[tempER.mktvalpct >= mktpct]

    ERtop10 = tempER.sort_values('Sharpe', ascending=False).head(20)
    ERbot10 = tempER.sort_values('Sharpe', ascending=True).head(20)

    result_dict = {
        'total_ER': ER
        , 'beta': Beta
        , 'index_ER': pd.DataFrame([['index', weighted_ER, mean_ER]], columns=['index', 'wER', 'ER'])
        , 'size_ER': pd.concat([size_ER,size_mean_ER], axis=1)
        , 'sector_ER': pd.concat([industry_name, industry_ER, industry_mean_ER, industry_beta], axis=1)
        , 'industry_ER': pd.concat([subindustry_name, subindustry_ER, subindustry_mean_ER, subindustry_beta], axis=1)
        , 'factor_ER': factor_ER
        , 'ERtop+bot10': pd.concat([ERtop10, ERbot10], axis=0)[['Name', 'sector', 'mktval','size', 'ER', 'Vol', 'Sharpe']]
    }

    weight_path = f'./output/0.export/{country}'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    #save each result in to csv file
    for key, value in result_dict.items():
        value.to_csv(f'./output/0.export/{country}/{key}.csv')

def run():
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Factor Research")

    # 2. Add arguments
    parser.add_argument("--c", nargs='+', default=['us','europe','japan','korea','china'], help="The list of countries")  # Positional
    parser.add_argument("--f", nargs='+', default=['BM', 'SMB', 'HML', 'MOM', 'OP', 'INV', 'REV'], help="The list of factors")  # Positional
    parser.add_argument("--d", default=None, type=str, help="The date of snapshot")
    parser.add_argument("--b", default='ME', type=str, choices=['ME','M'], help="Basis of frequency of calculation. If ME, the calculation is done based on month-end data. If M, the calculation is done based on monthly data.")

    # 3. Parse and use
    args = parser.parse_args()

    return args


start = None
if __name__ == '__main__':
    country_list = [
        ['us','RAY'],
        ['europe','SXXP'],
        ['japan','TPX500'],
        ['korea','KRX300'],
        ['china', 'MXCN']
    ]

    # initialize data
    for country, index in country_list:
        bpath = f'./cache/{country}'
        shutil.rmtree(bpath, ignore_errors=True)
        os.makedirs(bpath, exist_ok=True)

    # 1. Calculate Factor Index
    ## 1-1. Calculate Factor Index Weight
    print('Start Calculate Factor Index Weight')
    for country in country_list:
        calc_factor_weight(*country,
                           factorlist={'BM': 'amktcap'
                               , 'SMB': 'size'
                               , 'HML': 'value'
                               , 'MOM': 'momentum'
                               , 'OP': 'profitability'
                               , 'INV': 'investment'
                               , 'REV': 'reversal'
                                       }
                           )

    ## 1-2. Calculate Factor Index
    print('Start Calculate Factor Index')
    for country in country_list:
        calc_factor_index(*country, factorlist=['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV', 'REV'])

    # 2. Calculate Expected Return
    print('Calculate Expected Return')
    model_list = [
        ['us', 'RAY', 'ch4', 3, 12, ['bm', 'hml','smb','mom','op','inv']],
        ['japan', 'TPX500', 'capm', 3, 60, ['bm', 'hml','smb','mom','op','inv']],
        ['europe', 'SXXP', 'capm', 6, 12, ['bm', 'hml','smb','mom','op','inv']],
        ['korea', 'KRX300', 'capm', 3, 60, ['bm', 'hml', 'smb', 'mom', 'op', 'inv']],
        ['china', 'MXCN', 'ff3', 12, 60, ['bm', 'hml', 'smb', 'mom', 'op', 'inv']],
        ]

    for country_model in model_list:
        snapdate = calculate_Expected_Return(*country_model, end=None, calc_all=True)

    #snapdate = '2026-01-30'

    print('Calculate Formatted Expected return')
    model_list = [
        [snapdate, 'us', 'ch4', 3, 12, 0.7],
        [snapdate, 'japan', 'capm', 3, 60, -1],
        [snapdate, 'europe', 'capm', 6, 12, -1],
        [snapdate, 'korea', 'capm', 3, 60, -1],
        [snapdate, 'china', 'ff3', 12, 60, -1],
        ]
    for country_model in model_list:
        groupER(*country_model, factors=['BM', 'HML', 'INV', 'MOM', 'OP', 'SMB'])