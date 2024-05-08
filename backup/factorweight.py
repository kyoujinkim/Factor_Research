import pandas as pd
from src.funs import *
from src.calcWeight import WeightBuilder

country = 'korea'  # 'us' or 'korea' or 'japan' or 'europe'
rebal_freq = 'm'

# 1. 데이터 불러오기
# 1-1. 주간 시세 불러오기
price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
# 1-2. 시장 정보 불러오기d
market_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_market.csv', dtype=str, index_col=0, parse_dates=True)
# 1-3. 거래정지 여부 불러오기
halt_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
# 1-4. 시가총액 불러오기
mktval_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0,
                        parse_dates=True)
# 1-5. bookvalue(bps) 불러오기
bv_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_bookvalue.csv', dtype=float, index_col=0,
                    parse_dates=True)
# 1-6. earnings(eps) 불러오기
earn_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_earnings.csv', dtype=float, index_col=0,
                      parse_dates=True)
# 1-7. totalasset 불러오기
ta_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_totalasset.csv', dtype=float, index_col=0,
                    parse_dates=True)

# 데이터가 집계되지 않은 기간이 포함된 파일이 있으므로 해당 구간 제거
price_bd = price_bd.ffill().dropna(axis=0, how='all')
market_bd = market_bd.ffill().dropna(axis=0, how='all')
halt_bd = halt_bd.ffill().dropna(axis=0, how='all')
mktval_bd = mktval_bd.ffill().dropna(axis=0, how='all')
bv_bd = bv_bd.ffill().dropna(axis=0, how='all')
earn_bd = earn_bd.ffill().dropna(axis=0, how='all')
ta_bd = ta_bd.ffill().dropna(axis=0, how='all')

# date와 column 통일하기
basedate = findBaseDate(price_bd, market_bd, halt_bd, mktval_bd, bv_bd, earn_bd, ta_bd)
price_bd = price_bd[price_bd.index >= basedate]
market_bd = market_bd[market_bd.index >= basedate][price_bd.columns]
halt_bd = halt_bd[halt_bd.index >= basedate][price_bd.columns]
mktval_bd = mktval_bd[mktval_bd.index >= basedate][price_bd.columns]
bv_bd = bv_bd[bv_bd.index >= basedate][price_bd.columns]
earn_bd = earn_bd[earn_bd.index >= basedate][price_bd.columns]
ta_bd = ta_bd[ta_bd.index >= basedate][price_bd.columns]

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
bv_bd = bv_bd[bv_bd.index >= basedate]
earn_bd = earn_bd[earn_bd.index >= basedate]
ta_bd = ta_bd[ta_bd.index >= basedate]

startdate = price_bd.index[0] + pd.DateOffset(weeks=53)

if __name__ == "__main__":
    # BM portfolio 계산
    weighting_method = 'mktval'
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=-1, factor_top_n=-1
                       , price_bd=price_bd, market_bd=market_bd, halt_bd=halt_bd, mktval_bd=mktval_bd
                          )
    weight = wb.calcWeight(factor_bd=mktval_bd
                            , listed_window=52
                            , weighting_method=weighting_method
                            , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/BM+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')


    # Factor Quantile Portfolio 설정
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=-1, factor_top_n=0.2
                       , price_bd=price_bd, market_bd=market_bd, halt_bd=halt_bd, mktval_bd=mktval_bd
                       )
    # HML portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=bp_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/HML+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # SMB portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=-mktval_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/SMB+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # momentum portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=momentum_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/MOM+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # Operating Profitability portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=op_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/OP+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # Investment portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=inv_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=1)
    weight.to_csv(
        f'./weight/{country}/INV+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
    print('Process Finished')
