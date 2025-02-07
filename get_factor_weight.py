from glob import glob

import pandas as pd
from src.funs import *
from src.calcWeight import WeightBuilder
def calc_factor_weight(country, rebal_freq):

    # 1. 데이터 불러오기
    # 1-1. 주간 시세 불러오기
    price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-2. 시장 정보 불러오기d
    market_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_market.csv', dtype=str, index_col=0, parse_dates=True)
    # 1-3. 거래정지 여부 불러오기
    halt_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-4. 시가총액 불러오기
    mktval_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-5. bookvalue(bps) 불러오기
    bv_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_bookvalue.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-6. earnings(eps) 불러오기
    earn_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_earnings.csv', dtype=float, index_col=0, parse_dates=True)
    # 1-7. totalasset 불러오기
    ta_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_totalasset.csv', dtype=float, index_col=0, parse_dates=True)

    # 데이터가 집계되지 않은 기간이 포함된 파일이 있으므로 해당 구간 제거
    price_bd = price_bd.ffill().dropna(axis=0, thresh=100)
    market_bd = market_bd.ffill().dropna(axis=0, thresh=100)
    halt_bd = halt_bd.ffill().dropna(axis=0, thresh=100)
    mktval_bd = mktval_bd.ffill().dropna(axis=0, thresh=100)
    bv_bd = bv_bd.ffill().dropna(axis=0, thresh=100)
    earn_bd = earn_bd.ffill().dropna(axis=0, thresh=100)
    ta_bd = ta_bd.ffill().dropna(axis=0, thresh=100)

    # date와 column 통일하기
    basedate = findBaseDate(price_bd, market_bd, halt_bd, mktval_bd, bv_bd, earn_bd, ta_bd)
    price_bd = price_bd[price_bd.index >= basedate]
    market_bd = market_bd[market_bd.index >= basedate][price_bd.columns]
    halt_bd = halt_bd[halt_bd.index >= basedate][price_bd.columns]
    mktval_bd = mktval_bd[mktval_bd.index >= basedate][price_bd.columns]
    bv_bd = bv_bd[bv_bd.index >= basedate][price_bd.columns]
    earn_bd = earn_bd[earn_bd.index >= basedate][price_bd.columns]
    ta_bd = ta_bd[ta_bd.index >= basedate][price_bd.columns]

    # cai 불러오기
    cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
    cai_bd = cai_bd.reindex(price_bd.index, method='bfill').ffill()
    cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
    cai_bd = cai_bd.fillna(1)

    # cai를 기준으로 share관련 지표들 수정
    price_bd = price_bd * cai_bd
    bv_bd = bv_bd * cai_bd
    earn_bd = earn_bd * cai_bd

    # 팩터 계산
    momentum_bd = calculateMomentum(price_bd, LT_window=52, ST_window=4)
    bp_bd = calculateBP(price_bd, bv_bd)
    op_bd = calculateOP(earn_bd, bv_bd)
    inv_bd = calculateInv(ta_bd)

    # date 재통일
    basedate = findBaseDate(momentum_bd, bp_bd, op_bd, inv_bd)
    momentum_bd = momentum_bd[momentum_bd.index >= basedate]
    bp_bd = bp_bd[bp_bd.index >= basedate]
    op_bd = op_bd[op_bd.index >= basedate]
    inv_bd = inv_bd[inv_bd.index >= basedate]
    price_bd = price_bd[price_bd.index >= basedate]
    market_bd = market_bd[market_bd.index >= basedate]
    halt_bd = halt_bd[halt_bd.index >= basedate]
    mktval_bd = mktval_bd[mktval_bd.index >= basedate]

    startdate = price_bd.index[0] + pd.DateOffset(weeks=53)

    # BM portfolio 계산
    weighting_method = 'mktval'
    print(f'Calculating {country}+BM+{rebal_freq}')
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=-1, factor_top_n=-1
                       , price_bd=price_bd, market_bd=market_bd, halt_bd=halt_bd, mktval_bd=mktval_bd
                          )
    weight = wb.calcWeight(factor_bd=mktval_bd
                            , listed_window=52
                            , weighting_method=weighting_method
                            , weight_limit=0.3
                            , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/BM.csv')

    # Factor Quantile Portfolio 설정
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=-1, factor_top_n=0.2
                       , price_bd=price_bd, market_bd=market_bd, halt_bd=halt_bd, mktval_bd=mktval_bd
                       )

    weighting_method = 'equal+LS'

    # HML portfolio 계산
    print(f'Calculating {country}+HML+{rebal_freq}')
    weight = wb.calcWeight(factor_bd=bp_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/HML.csv')

    # SMB portfolio 계산
    print(f'Calculating {country}+SMB+{rebal_freq}')
    weight = wb.calcWeight(factor_bd=-mktval_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/SMB.csv')

    # momentum portfolio 계산
    print(f'Calculating {country}+MOM+{rebal_freq}')
    weight = wb.calcWeight(factor_bd=momentum_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/MOM.csv')

    # Operating Profitability portfolio 계산
    print(f'Calculating {country}+OP+{rebal_freq}')
    weight = wb.calcWeight(factor_bd=op_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/OP.csv')

    # Investment portfolio 계산
    print(f'Calculating {country}+INV+{rebal_freq}')
    weight = wb.calcWeight(factor_bd=-inv_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/factor_weight/INV.csv')
    print('Process Finished')

def calc_mvo_weight(snapdate, country, rebal_freq, model, avgwindow, datawindow, weightlimit=0.1):
    # 1. 데이터 불러오기
    # 1-1. 주간 시세 불러오기
    price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0,
                           parse_dates=True)
    if country == 'us':  # drop game stop equity, since it's too volatile
        price_bd = price_bd.drop('US36467W1099', axis=1)
    # 1-2. 시장 정보 불러오기
    market_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_market.csv', dtype=str, index_col=0,
                            parse_dates=True)
    # 1-3. 거래정지 여부 불러오기
    halt_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0,
                          parse_dates=True)
    # 1-4. 시가총액 불러오기
    mktval_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0,
                            parse_dates=True)
    # cai 불러오기
    cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
    cai_bd = cai_bd.reindex(price_bd.index, method='bfill').ffill()
    cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
    cai_bd = cai_bd.fillna(1)

    # cai를 기준으로 share관련 지표들 수정
    price_bd *= cai_bd

    path = f'./output/{country}/Expected_Return/{model}_{avgwindow}m_{datawindow}m_snapshot{snapdate}.csv'

    name = path.split('\\')[-1].split('/')[-1].split('_snapshot')[0]
    # 2. 예상수익률 불러오기
    # pred = get_exp_ret(path)
    pred = pd.read_csv(path, index_col=0).T
    pred.index = pd.to_datetime(pred.index)

    # 데이터가 집계되지 않은 기간이 포함된 파일이 있으므로 해당 구간 제거
    price = price_bd.ffill().dropna(axis=0, how='all')
    market = market_bd.ffill().dropna(axis=0, how='all')
    halt = halt_bd.ffill().dropna(axis=0, how='all')
    mktval = mktval_bd.ffill().dropna(axis=0, how='all')

    # date와 column 통일하기
    basedate = findBaseDate(price, market, halt, mktval)
    price = price[price.index >= basedate]
    market = market[market.index >= basedate][price.columns]
    halt = halt[halt.index >= basedate][price.columns]
    mktval = mktval[mktval.index >= basedate][price.columns]

    startdate =pred.index[0]

    weight_path = f'./weight/{country}/mvo_weight'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # BM portfolio 계산
    print(f'Calculating {country}+{name}+{rebal_freq}')

    rm = 'MV'

    # declare weighting method
    weighting_method = 'optmvo'

    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=0.95, factor_top_n=[10, 0.2, 200]
                       , price_bd=price, market_bd=market, halt_bd=halt, mktval_bd=mktval
                       )
    #set current date as date_list
    wb.date_list = [startdate]

    weight = wb.calcWeight(factor_bd=pred
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1
                           , weight_limit=weightlimit
                           , rm=rm
                           , active_multiproc=False
                           )

    weight.to_csv(
        f'{weight_path}/{name}+{weighting_method}+{rm}+{rebal_freq}.csv')
