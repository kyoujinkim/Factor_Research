from glob import glob

import pandas as pd
from pandas._libs.tslibs.offsets import MonthEnd

from src.funs import *
from src.calcWeight import WeightBuilder


if __name__ == "__main__":

    country = 'us'  # 'us' or 'korea' or 'japan' or 'europe'
    rebal_freq = 'm'

    # 1. 데이터 불러오기
    # 1-1. 주간 시세 불러오기
    price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    if country == 'us': # drop game stop equity, since it's too volatile
        price_bd = price_bd.drop('US36467W1099', axis=1)
    # 1-2. 시장 정보 불러오기
    market_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_market.csv', dtype=str, index_col=0, parse_dates=True)
    # 1-3. 거래정지 여부 불러오기
    halt_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-4. 시가총액 불러오기
    mktval_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-5. 산업분류 불러오기
    industry_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)

    ERpath = glob(f'./output/{country}/Expected_Return/ff6*.pkl')
    for path in ERpath:
        name = path.split('\\')[-1].replace('.pkl','').replace('_','+')
        # 2. 예상수익률 불러오기
        #pred = get_exp_ret(path)
        pred = pd.concat(load_pickle(path), axis=1).T

        # 2-2. 비교할 실제 수익률 불러오기
        price_m = price_bd.resample('M').last()
        ret = price_m.pct_change().dropna(axis=0, how='all')

        # 데이터가 집계되지 않은 기간이 포함된 파일이 있으므로 해당 구간 제거
        price = price_bd.ffill().dropna(axis=0, how='all')
        market = market_bd.ffill().dropna(axis=0, how='all')
        halt = halt_bd.ffill().dropna(axis=0, how='all')
        mktval = mktval_bd.ffill().dropna(axis=0, how='all')
        pred = pred.dropna(axis=0, how='all')

        # date와 column 통일하기
        basedate = findBaseDate(price, market, halt, mktval, pred, ret)
        price = price[price.index >= basedate]
        market = market[market.index >= basedate][price.columns]
        halt = halt[halt.index >= basedate][price.columns]
        mktval = mktval[mktval.index >= basedate][price.columns]
        # pred_bd는 이미 column이 필터링되어 있으므로 skip
        pred = pred[pred.index >= basedate]
        ret = ret[ret.index >= basedate][price.columns]

        startdate = price.index[0] + pd.DateOffset(weeks=53)

        for rm in ['MV']:#, 'MDD', 'CVaR']:
            # BM portfolio 계산
            print(f'Calculating {country}+{name}+{rm}+{rebal_freq}')

            # declare weighting method
            weighting_method = 'optmvo'

            # if path exists, skip
            if len(glob(f'./weight/{country}/{name}+{weighting_method}+{rm}+{rebal_freq}_*.csv'))>0:
                continue

            wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                               , mktval_top_n=-1, factor_top_n=200
                               , price_bd=price, market_bd=market, halt_bd=halt, mktval_bd=mktval
                               )
            weight = wb.calcWeight(factor_bd=pred
                                   , listed_window=52
                                   , weighting_method=weighting_method
                                   , multiproc_partition=1
                                   , weight_limit=0.1
                                   , rm=rm
                                   )
            weight.to_csv(
                f'./weight/{country}/{name}+{weighting_method}+{rm}+{rebal_freq}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
