import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

import config
from src.loadData import DataProcessor
from src.calcWeight_2v import WeightBuilder
from src.calcER_2v import Calculator
from src.backtest import plan_factor_index, run_factor_index


def _open_dp(index=None, stocks=None):
    '''DataProcessor with cache + raw store on, loaded with the members of index (or the given stocks)'''
    dp = DataProcessor()
    dp.turnon_cache()
    dp.set_raw_store()
    index_master = None
    if index is not None:
        # index별로 저장되어있는 index_master 불러오기
        index_master = pd.read_parquet(f'{DataProcessor.dbpath}/members/master_{index}.parquet', engine='pyarrow')
        stocks = index_master.columns
    dp.load_stocks(stocks)
    return dp, index_master


def calc_factor_weight(country, index, factorlist, recalc_w=False):

    if not os.path.exists(f'./weight/{country}/factor_weight'):
        os.makedirs(f'./weight/{country}/factor_weight')

    bpath = f'./cache/{country}'

    dp, index_master = _open_dp(index)

    for key in factorlist.keys():

        factor = factorlist[key]
        data_list = ['amktcap', 'returns']
        data_list.append(factor)

        if recalc_w:
            w_prev = pd.DataFrame()
            w_end = None
        else:
            w_prev = pd.read_csv(f'./weight/{country}/factor_weight/{key}.csv', index_col=0, parse_dates=True).iloc[:-24]
            w_end = (w_prev.index[-1]-relativedelta(weeks=5)).strftime('%Y-%m-%d')

        # build data
        for data in tqdm(data_list):
            dp.build_data(bpath, data, w_end, None)
        # the weights read the parquet files just written; free the in-memory intermediates (GBs for a full history)
        # before the workers start
        dp.turnoff_cache()
        dp.turnon_cache()

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

def calc_factor_index(country, index, factorlist, full=False):
    '''
    factor index = backtest of the factor weights. Only the tail from the first new/changed rebalance date is
    backtested and spliced onto the saved series (see src/backtest.py). full=True rebuilds everything, e.g. after
    re-downloading a stock's price/dividend history.
    '''

    if not os.path.exists(f'./output/{country}/bm'):
        os.makedirs(f'./output/{country}/bm')

    bpath = f'./cache/{country}'

    dp, index_master = _open_dp(index)

    # load price (full history: it is also the ER stage's input)
    dp.build_data(bpath, 'zprice', None, None)
    p = pd.read_parquet(f'{bpath}/zprice.parquet', engine='pyarrow')

    '''load weight, decide where each factor's backtest restarts'''
    weights, plans = {}, {}
    for factor in factorlist:
        if factor != 'HS':
            weights[factor] = pd.read_csv(f'./weight/{country}/factor_weight/{factor}.csv', index_col=0, parse_dates=True)
            plans[factor] = plan_factor_index(weights[factor], p.columns, f'./output/{country}/bm/{factor}.csv', thresh=30, full=full)

    # load div: only from the earliest restart when every factor runs incrementally
    restarts = [pl['restart'] for pl in plans.values()]
    zstart = min(restarts).strftime('%Y-%m-%d') if restarts and all(r is not None for r in restarts) else None
    dp.build_data(bpath, 'zdps', zstart, None)
    d = pd.read_parquet(f'{bpath}/zdps.parquet', engine='pyarrow')
    if not set(p.columns) <= set(d.columns):
        # the weight hashes assume every price column has coupons; otherwise rebuild everything
        dp.build_data(bpath, 'zdps', None, None)
        d = pd.read_parquet(f'{bpath}/zdps.parquet', engine='pyarrow')
        plans = {f: plan_factor_index(weights[f], p.columns, f'./output/{country}/bm/{f}.csv', thresh=30, full=True) for f in plans}

    factor_ts = []
    for factor in factorlist:
        if factor == 'HS':
            res = pd.read_csv(f'{DataProcessor.dbpath}/external_factor/{country}/hs_delta.csv', index_col=0, parse_dates=True)[['HS_Delta']]
            res.columns = [factor]
            res.to_csv(f'./output/{country}/bm/{factor}.csv')
        else:
            weight = weights[factor]
            out_csv = f'./output/{country}/bm/{factor}.csv'
            plan = plans[factor]
            t = time.perf_counter()
            res = run_factor_index(factor, weight, p, d, out_csv, plan, thresh=30)
            print(f'[factor index] {country} {factor}: {"incremental" if plan["restart"] is not None else "full"} '
                  f'({plan["reason"]}) {time.perf_counter()-t:.0f}s')
        factor_ts.append(res)

    factor_ts = pd.concat(factor_ts, axis=1).ffill()
    os.makedirs(f'output/0.export/{country}', exist_ok=True)
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

    dp, index_master = _open_dp(index)

    for data in tqdm(['zprice']):
        dp.build_data(f'./cache/{country}', data, None, None)

    # load price
    p = pd.read_parquet(f'./cache/{country}/zprice.parquet', engine='pyarrow')
    rf = pd.read_excel(f'{DataProcessor.dbpath}/rf.xlsx', sheet_name='rf', index_col=0, parse_dates=True)[[country]].iloc[1:].astype(float)
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

    dp, _ = _open_dp(stocks=ER.index)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Factor Research")
    parser.add_argument("--c", nargs='+', default=list(config.MARKETS), choices=list(config.MARKETS), help="countries to run")
    parser.add_argument("--f", nargs='+', default=list(config.FACTOR_SCORES), choices=list(config.FACTOR_SCORES), help="factors whose weights/indices are rebuilt")
    parser.add_argument("--d", default=None, type=str, help="snapshot end date of the expected return (default: latest)")
    parser.add_argument("--recalc-weight", action='store_true', help="rebuild the factor weights from the start instead of the last 24 rows")
    parser.add_argument("--rebuild-index", action='store_true', help="backtest the factor indices from the start (after repairing a stock's price/dividend history)")
    parser.add_argument("--skip-er-history", action='store_true', help="only the ER snapshot, not the full ER time series (.pkl)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    markets = {c: config.MARKETS[c] for c in args.c}

    # 1. Calculate Factor Index
    ## 1-1. Calculate Factor Index Weight
    print('Start Calculate Factor Index Weight')
    for country, m in markets.items():
        calc_factor_weight(country, m['index'],
                           factorlist={f: config.FACTOR_SCORES[f] for f in args.f},
                           recalc_w=args.recalc_weight)

    ## 1-2. Calculate Factor Index
    # The factor index is recomputed only from the first new/changed rebalance date (see calc_factor_index).
    # Every factor is listed so factor_hs.csv stays complete; unchanged ones are skipped by their saved state.
    print('Start Calculate Factor Index')
    for country, m in markets.items():
        calc_factor_index(country, m['index'], factorlist=list(config.FACTOR_SCORES), full=args.rebuild_index)

    # 2. Calculate Expected Return
    print('Calculate Expected Return')
    snapdates = {}
    for country, m in markets.items():
        snapdates[country] = calculate_Expected_Return(country, m['index'], m['model'], m['avg_window'], m['window'],
                                                       config.ER_FACTORS, end=args.d, calc_all=not args.skip_er_history)

    print('Calculate Formatted Expected return')
    for country, m in markets.items():
        groupER(snapdates[country], country, m['model'], m['avg_window'], m['window'], m['mktpct'], factors=config.EXPORT_FACTORS)
