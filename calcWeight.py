# import custom functions
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
        :param mktval_top_n: 시가총액 상위 N개 종목
        :param factor_top_n: 팩터 상위 N개 종목
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
        current_date, factor_df, factor_window, weighting_method, market_type, mktval_top_n, factor_top_n, weight_limit = var
        # 상장된 지 1년 이상된 종목만 필터링
        price_df = price_bd[price_bd.index <= current_date][-factor_window:]
        code_list = filterByDate(price_df, price_df.columns.tolist(), window=factor_window)

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
        code_list = filterByFactor(factor_df, code_list, top_n=factor_top_n)

        #주도주를 포함하는게 좋은지 확인하기 위해서 임의로 주도주를 포함하는 프로세스를 추가한다(about to be deleted in future)
        #code_list = list(set(code_list + leader))

        # 3. Weight 계산
        # 3-1. 시가총액 비중 계산
        if weighting_method == 'mktval':
            weight = weightMarktValue(code_list, mktval_df)
        # 3-2. 동일가중 비중 계산
        elif weighting_method == 'equal':
            weight = weightEqual(code_list, mktval_df)
        # 방법론 중에 없는 경우 경고 표시후 종료
        elif weighting_method == 'factor':
            weight = weightFactor(code_list, factor_df)
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
                   , factor_window: int = 52, weighting_method: str = 'equal', weight_limit: float = None
                   , multiproc_partition: int = 1):
        '''
        비중값을 계산하는 함수
        :param factor_bd: 팩터 데이터(데이터프레임)
        :param factor: 어떤 팩터를 기준으로 비중을 계산할 것인지 지정(e.g. momentun, value, quality, size, etc.)
        :param factor_window: 팩터를 계산하는 기간 지정
        :param weighting_method: 어떤 방식으로 비중을 계산할 것인지 지정(e.g. equal, mktval, factor, etc.)
        :param multiproc_partition: 멀티프로세싱을 위한 파티션 수(메모리 overflow 방지)
        :return: 비중값 데이터프레임
        '''
        # should wrap up self variables to use multiprocessing in proper way
        # date_list, factor_df, factor_window, weighting_method, market_type, mktval_top_n, factor_top_n
        # for each date in date_list wrap up the variables above
        total_weight = []
        chunk_count = 0
        for date_chunk in self.chunk_date(self.date_list, int(len(self.date_list)/multiproc_partition)):
            chunk_count += 1
            print(f'Align Variable : {chunk_count}/{multiproc_partition} Partition')
            var_list = []
            for date in date_chunk:
                factor_df = factor_bd[factor_bd.index <= date].iloc[-1]
                var = [date, factor_df, factor_window, weighting_method, self.market_type, self.mktval_top_n, self.factor_top_n, weight_limit]
                var_list.append(var)
            weight = process_map(self.makeWeight
                                 , var_list
                                 , max_workers=os.cpu_count()-1
                                 , desc=f'calc weight : {chunk_count}/{multiproc_partition} Partition')
            total_weight.extend(weight)

        return pd.concat(weight)

country = 'korea' #'us' or 'korea'
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
# 외국인 지분 계산
forsh_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_foreign_share.csv', dtype=float, index_col=0, parse_dates=True)


startdate = price_bd.index[0] + pd.DateOffset(weeks=53)
startdate = forsh_bd.index[0] + pd.DateOffset(weeks=53)
#startdate = '2020-03-30'
#startdate = '1996-03-29'

if __name__ == "__main__":

    # 1-5. 모멘텀 팩터 계산
    # momentum_bd = calculateMomentum(price_bd, LT_window=52, ST_window=4)
    # lowVol_bd = calculateLowVol(price_bd, return_window=1, std_window=52)
    fscoeff_bd = calculateFSCoeff(price_bd, forsh_bd, window=52)

    #우선 필요한 거
    #시가총액별로 상위 1/3, 중위 1/3, 하위 1/3 갈랐을 때 동일가중 모멘텀 포트폴리오, 시가총액가중 모멘텀 포트폴리오
    #동일가중 시장 포트폴리오, 시가총액가중 시장 포트폴리오
    #한국 / 미국
    #포트폴리오 백테스트 결과 필요함

    '''weighting_method = 'mktval'
    wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                       , mktval_top_n=0.75, factor_top_n=-1)

    weight = wb.calcWeight(factor_bd=mktval_bd
                           , factor_window=52
                           , weighting_method=weighting_method
                           , multiproc_partition=2)#, weight_limit=weight_limit)
    weight.to_csv(
        f'./weight/{country}/bm+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')'''
    for factor_top_n in [10, 20, 50, 100]:
        weighting_method = 'equal'
        wb = WeightBuilder(rebal_freq=rebal_freq, startdate=startdate, enddate=None
                           , mktval_top_n=0.75, factor_top_n=50)

        weight = wb.calcWeight(factor_bd=fscoeff_bd
                               , factor_window=52
                               , weighting_method=weighting_method
                               , multiproc_partition=2)
        weight.to_csv(
            f'./weight/{country}/FScoeff{factor_top_n}+{weighting_method}_{weight.index[0].strftime("%Y-%m-%d")}_{price_bd.index[-1].strftime("%Y-%m-%d")}.csv')


    print('Process Finished')
