# import custom functions
import math

from pandas._libs.tslibs.offsets import MonthEnd

from src.funs import *


class WeightBuilder:
    def __init__(self
                 , rebal_freq='m', startdate=None, enddate=None
                 , market_type: str = None, mktval_top_n=500, factor_top_n=50
                 , price_bd=None, market_bd=None, halt_bd=None, mktval_bd=None, industry_bd=None):
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
        self.price_bd = price_bd
        self.market_bd = market_bd
        self.halt_bd = halt_bd
        self.mktval_bd = mktval_bd
        self.industry_bd = industry_bd

        if startdate is None:
            startdate = price_bd.index[0]
        if enddate is None:
            enddate = price_bd.index[-1]
        # 0. Weight을 만들기 위한 리밸런싱 시점 결정
        self.date_list = pd.period_range(start=startdate, end=enddate, freq=rebal_freq).to_timestamp()[1:]

    def chunk_date(self, date_list, chunk_size):
        for i in range(0, len(date_list), chunk_size):
            yield date_list[i:i + chunk_size]

    def calcWeight(self, factor_bd
                   , listed_window: int = 52
                   , weighting_method: str = 'equal'
                   , multiproc_partition: int = 1
                   , **kwargs):
        '''
        비중값을 계산하는 함수
        :param factor_bd: 팩터 데이터(데이터프레임)
        :param factor: 어떤 팩터를 기준으로 비중을 계산할 것인지 지정(e.g. momentun, value, quality, size, etc.)
        :param listed_window: 종목이 상장되었던 최소 기간 기준
        :param weighting_method: 어떤 방식으로 비중을 계산할 것인지 지정(e.g. equal, mktval, factor, etc.)
        :param multiproc_partition: 멀티프로세싱을 위한 파티션 수(메모리 overflow 방지)
        :param kwargs: 기타 인자
        /weight_limit: float: 개별 종목의 비중 상한선
        /rm: str: risk measure(e.g. MV, MDD. CVaR, etc.) - https://riskfolio-lib.readthedocs.io/
        /industry: str: 선택하려는 산업 (클래스 선언부에서 industry_bd가 None이 아닌 경우에만 동작함)
        /workers: int: 멀티프로세싱을 위한 워커 수
        :return: 비중값 데이터프레임
        '''
        weight_limit = kwargs.get('weight_limit', None)
        rm = kwargs.get('rm', 'MV')
        industry = kwargs.get('industry', None)
        workers = kwargs.get('workers', os.cpu_count()-1)

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
                price_df = self.price_bd[self.price_bd.index <= date][-listed_window:].copy()
                rtn_df = self.price_bd.pct_change()[self.price_bd.index <= date][-listed_window:].copy()

                # 현재 시점의 데이터만 필터링
                if isinstance(self.market_bd, type(None)):
                    market_df = None
                else:
                    market_df = self.market_bd[self.market_bd.index <= date].iloc[-1].copy()
                if isinstance(self.halt_bd, type(None)):
                    halt_df = None
                else:
                    halt_df = self.halt_bd[self.halt_bd.index <= date].iloc[-1].copy()
                if isinstance(self.mktval_bd, type(None)):
                    mktval_df = None
                else:
                    mktval_df = self.mktval_bd[self.mktval_bd.index <= date].iloc[-1].copy()
                if isinstance(self.industry_bd, type(None)):
                    industry_df = None
                else:
                    industry_df = self.industry_bd[self.industry_bd.index <= date].iloc[-1].copy()

                var = [date
                    , factor_df, price_df, rtn_df, market_df, halt_df, mktval_df, industry_df
                    , listed_window, weighting_method, self.market_type, self.mktval_top_n, self.factor_top_n
                    , weight_limit, rm, industry]
                var_list.append(var)

            weight = process_map(makeWeight
                                 , var_list
                                 , max_workers=workers
                                 , desc=f'calc weight : {chunk_count}/{multiproc_partition} Partition')

            total_weight.extend(weight)

        return pd.concat(total_weight)

def makeWeight(var):
    (current_date
     , factor_df, price_df, rtn_df, market_df, halt_df, mktval_df, industry_df
     , listed_window, weighting_method, market_type, mktval_top_n, factor_top_n
     , weight_limit, rm, industry) = var

    # 상장된 지 1년 이상된 종목만 필터링
    code_list = filterByDate(price_df, price_df.columns.tolist(), window=listed_window)

    # 2. 필터링
    # 2-1. KOSPI/KOSDAQ 구분
    code_list = filterByMarket(market_df, code_list, market_type=market_type)
    # 2-2. Sector 구분
    code_list = filterByIndustry(industry_df, code_list, industry)
    # 2-3. 거래정지된 종목 제외
    code_list = filterByHalt(halt_df, code_list)
    # 2-4. 시가총액 상위 N개 종목
    code_list = filterByMarketValue(mktval_df, code_list, top_n=mktval_top_n)
    # 2-5. 팩터가 가장 높은 상위 N개 종목
    if 'LS' in weighting_method:
        code_list_long = filterByFactor(factor_df, code_list, top_n=factor_top_n)
        code_list_short = filterByFactor(factor_df, code_list, top_n=factor_top_n, short=True)
    else:
        code_list = filterByFactor(factor_df, code_list, top_n=factor_top_n)

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
    # Mean-Variance 최적화 기반 비중 계산
    elif weighting_method == 'optmvo':
        weight = weightMVO(code_list, rtn_df, factor_df, rm)
    # 방법론 중에 없는 경우 경고 표시후 종료
    else:
        warnings.warn('해당 방법론은 없습니다. 다시 입력해주세요.')
        return None
    weight = pd.DataFrame(weight).T

    if type(weight_limit) == float:
        weight = limitWeight(weight, limit=weight_limit)

    return weight
