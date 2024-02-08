# import custom functions
import math

from funs import *


class WeightBuilder:
    def __init__(self
                 , rebal_freq='m', startdate=None, enddate=None
                 , market_type: str = None, mktval_top_n=500, factor_top_n=50):
        '''
        비중값을 계산하기 위한 클래스
        :param price_bd: 가격 데이터
        :param market_bd: 시장 정보 데이터
        :param halt_bd: 거래정지 여부 데이터
        :param mktval_bd: 시가총액 데이터
        :param rebal_freq: 리밸런싱 주기
        :param startdate: 테스트 시작일
        :param enddate: 테스트 종류일
        :param market_type: KOSPI/KOSDAQ
        :param mktval_top_n: 시가총액 상위 N개 종목 (list, int or float)
        :param factor_top_n: 팩터 상위 N개 종목 (list, int or float)
        '''

        self.market_type = market_type
        self.mktval_top_n = mktval_top_n
        self.factor_top_n = factor_top_n
        global price_bd

        if startdate is None:
            startdate = price_bd.index[0]
        if enddate is None:
            enddate = price_bd.index[-1]
        # 0. Weight을 만들기 위한 리밸런싱 시점 결정
        self.date_list = pd.period_range(start=startdate, end=enddate, freq=rebal_freq).to_timestamp()[1:]

    def makeWeight(self, var):
        global price_bd, market_bd, halt_bd, mktval_bd
        current_date, factor_df, listed_window, weighting_method, market_type, mktval_top_n, factor_top_n, weight_limit = var
        # 상장된 지 1년 이상된 종목만 필터링
        price_df = price_bd[price_bd.index <= current_date][-listed_window:]
        code_list = filterByDate(price_df, price_df.columns.tolist(), window=listed_window)

        # 현재 시점의 데이터만 필터링
        market_df = market_bd[market_bd.index <= current_date].iloc[-1]
        halt_df = halt_bd[halt_bd.index <= current_date].iloc[-1]
        mktval_df = mktval_bd[mktval_bd.index <= current_date].iloc[-1]

        # 2. 필터링
        # 2-1. KOSPI/KOSDAQ 구분
        code_list = filterByMarket(market_df, code_list, market_type=market_type)
        # 2-2. 거래정지된 종목 제외
        code_list = filterByHalt(halt_df, code_list)
        # 2-3. 시가총액 상위 N개 종목
        code_list = filterByMarketValue(mktval_df, code_list, top_n=mktval_top_n)
        # 주도주
        #leader = code_list[:5]
        # 2-4. 팩터가 가장 높은 상위 N개 종목
        if 'LS' in weighting_method:
            factor_top_n_long = factor_top_n
            code_list_long = filterByFactor(factor_df, code_list, top_n=factor_top_n)
            code_list_short = filterByFactor(factor_df, code_list, top_n=factor_top_n, short=True)
        else:
            code_list = filterByFactor(factor_df, code_list, top_n=factor_top_n)

        #주도주를 포함하는게 좋은지 확인하기 위해서 임의로 주도주를 포함하는 프로세스를 추가한다(about to be deleted in future)
        #code_list = list(set(code_list + leader))

        # 3. Weight 계산
        # 3-1. 시가총액 비중 계산
        if weighting_method == 'mktval':
            weight = weightMarketValue(code_list, mktval_df)
        # 3-2. 동일가중 비중 계산
        elif weighting_method == 'equal':
            weight = weightEqual(code_list, mktval_df)
        elif weighting_method == 'factor':
            weight = weightFactor(code_list, factor_df)
        # 롱숏이 추가된 동일가중 비중 계산
        elif weighting_method == 'equal+LS':
            weight_long = weightEqual(code_list_long, mktval_df)
            weight_short = weightEqual(code_list_short, mktval_df) * -1
            weight = pd.concat([weight_long, weight_short], axis=0)
        # 방법론 중에 없는 경우 경고 표시후 종료
        else:
            warnings.warn('해당 방법론은 없습니다. 다시 입력해주세요.')
            return None
        weight = pd.DataFrame(weight).T

        if type(weight_limit) == float:
            weight = limitWeight(weight, limit=weight_limit)

        return weight

    def chunk_date(self, date_list, chunk_size):
        for i in range(0, len(date_list), chunk_size):
            yield date_list[i:i + chunk_size]

    def calcWeight(self, factor_bd
                   , listed_window: int = 52, weighting_method: str = 'equal', weight_limit: float = None
                   , multiproc_partition: int = 1):
        '''
        비중값을 계산하는 함수
        :param factor_bd: 팩터 데이터(데이터프레임)
        :param factor: 어떤 팩터를 기준으로 비중을 계산할 것인지 지정(e.g. momentun, value, quality, size, etc.)
        :param listed_window: 종목이 상장되었던 최소 기간 기준
        :param weighting_method: 어떤 방식으로 비중을 계산할 것인지 지정(e.g. equal, mktval, factor, etc.)
        :param multiproc_partition: 멀티프로세싱을 위한 파티션 수(메모리 overflow 방지)
        :return: 비중값 데이터프레임
        '''
        # should wrap up self variables to use multiprocessing in proper way
        # date_list, factor_df, factor_window, weighting_method, market_type, mktval_top_n, factor_top_n
        # for each date in date_list wrap up the variables above
        total_weight = []
        chunk_count = 0
        for date_chunk in self.chunk_date(self.date_list, math.ceil(len(self.date_list) / multiproc_partition)):
            chunk_count += 1
            print(f'Align Variable : {chunk_count}/{multiproc_partition} Partition')
            var_list = []
            for date in date_chunk:
                factor_df = factor_bd[factor_bd.index <= date].iloc[-1]
                var = [date, factor_df, listed_window, weighting_method, self.market_type, self.mktval_top_n, self.factor_top_n, weight_limit]
                var_list.append(var)

            weight = process_map(self.makeWeight
                                 , var_list
                                 , max_workers=os.cpu_count()-1
                                 , desc=f'calc weight : {chunk_count}/{multiproc_partition} Partition')
            total_weight.extend(weight)

        return pd.concat(total_weight)

country = 'korea' #'us' or 'korea' or 'japan' or 'europe'
rebal_freq = 'm'

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

    '''# BM portfolio 계산
    weighting_method = 'mktval'
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                          , mktval_top_n=-1, factor_top_n=-1
                          )
    weight = wb.calcWeight(factor_bd=mktval_bd
                            , listed_window=52
                            , weighting_method=weighting_method
                            , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/BM+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
    '''

    # Factor Quantile Portfolio 설정
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                            , mktval_top_n=-1, factor_top_n=0.2
                            )
    # HML portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=bp_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/HML+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # SMB portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=-mktval_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/SMB+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
    
    # momentum portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=momentum_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/MOM+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # Operating Profitability portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=op_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/OP+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')

    # Investment portfolio 계산
    weighting_method = 'equal+LS'
    weight = wb.calcWeight(factor_bd=inv_bd
                           , listed_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)
    weight.to_csv(
        f'./weight/{country}/INV+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')
    print('Process Finished')
