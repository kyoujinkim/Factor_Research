import pandas as pd
from src.funs import *
from src.calcWeight import WeightBuilder

country = 'europe'  # 'us' or 'korea' or 'japan' or 'europe'
rebal_freq = 'm'

# 1. 데이터 불러오기
# 1-1. 주간 시세 불러오기
price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0,
                       parse_dates=True)
# 1-2. 시장 정보 불러오기
market_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_market.csv', dtype=str, index_col=0,
                        parse_dates=True)
# 1-3. 거래정지 여부 불러오기
halt_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0,
                      parse_dates=True)
# 1-4. 시가총액 불러오기
mktval_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0,
                        parse_dates=True)
# 1-5. 산업분류 불러오기
ind_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0,
                        parse_dates=True)

# 데이터가 집계되지 않은 기간이 포함된 파일이 있으므로 해당 구간 제거
price_bd = price_bd.ffill().dropna(axis=0, how='all')
market_bd = market_bd.ffill().dropna(axis=0, how='all')
halt_bd = halt_bd.ffill().dropna(axis=0, how='all')
mktval_bd = mktval_bd.ffill().dropna(axis=0, how='all')
#ind_df의 기간을 price_bd에 맞추기
ind_bd = ind_bd.reindex(price_bd.index).ffill().bfill().dropna(axis=0, how='all')

# date와 column 통일하기
basedate = findBaseDate(price_bd, market_bd, halt_bd, mktval_bd, ind_bd)
price_bd = price_bd[price_bd.index >= basedate]
market_bd = market_bd[market_bd.index >= basedate][price_bd.columns]
halt_bd = halt_bd[halt_bd.index >= basedate][price_bd.columns]
mktval_bd = mktval_bd[mktval_bd.index >= basedate][price_bd.columns]
ind_bd = ind_bd[ind_bd.index >= basedate]

startdate = price_bd.index[0] + pd.DateOffset(weeks=53)

# set mapped index of industry
sector_class_len = 2
map_ind = pd.read_csv(f'D:/Factor_DB/{country}/{country}_industrymap.csv', dtype=str, index_col=0)
map_ind.index = map_ind.index.map(str)
unique_ind = ind_bd.iloc[-1].str[:sector_class_len].drop_duplicates().dropna().unique()
ind_list = map_ind.loc[unique_ind]

if __name__ == "__main__":
    # industry portfolio 계산
    weighting_method = 'mktval'
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=-1, factor_top_n=-1
                       , price_bd=price_bd, market_bd=market_bd, halt_bd=halt_bd, mktval_bd=mktval_bd
                       , industry_bd=ind_bd
                       )

    for ind in ind_list.index:
        weight = wb.calcWeight(factor_bd=mktval_bd
                               , listed_window=52
                               , weighting_method=weighting_method
                               , industry=ind
                               , multiproc_partition=1
                               , workers=4)
        weight.to_csv(
            f'./weight/{country}/sector{ind}+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
