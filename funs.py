#import packages
import os

import numpy as np
import pandas as pd
from datetime import datetime as dt

#turn off future warning
import warnings

from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

'''
목표: 한국 시장에서 모멘텀 포트의 성과 구하기
따라서 우선 모멘텀 포트를 구성하기 위한 모멘텀 팩터를 계산해야 함
모멘텀 팩터는 12개월 수익률과 1개월 수익률의 차이로 계산
모멘텀 포트는 다음과 같은 필터링 Function을 통해 구성
입력값은 데이터 담고 있는 DataFrame과 종목코드
1.KOSPI/KOSDAQ 구분
2.거래정지된 종목 제외
3.시가총액 상위 N개 종목
4.모멘텀 팩터가 가장 높은 상위 N개 종목
'''

# 상장된지 1년된 종목만 필터링
def filterByDate(price_df, code_list:list, window:int=52):
    '''
    상장된지 1년된 종목만 필터링
    :param price_df: 가격 데이터
    :param code_list: 종목코드
    :param window: 상장된지 window주 이상된 종목만 필터링
    :return:
    '''
    # 1. 상장된 지 n주 이상된 종목만 필터링
    price_df = price_df.replace(0, np.nan)
    filtered_df = price_df[code_list].dropna(axis=1, thresh=window)

    return filtered_df.columns.tolist()

def filterByMarket(market_df, code_list:list, market_type:str=None):
    '''
    KOSPI/KOSDAQ 구분
    :param market_df: 상장된 시장정보
    :param code_list: 종목코드
    만약 market_type이 KOSPI이면 KOSPI 종목만 필터링
    만약 market_type이 KOSDAQ이면 KOSDAQ 종목만 필터링
    만약 market_type이 None이면 KOSPI/KOSDAQ 모두 필터링
    :return:
    '''
    # 1. market_df에서 code_list에 해당하는 종목만 필터링
    market_df = market_df[code_list]
    # 2. 반드시 필터링해야하는 Exchange 제외
    market_df = market_df[market_df != '외감']
    market_df = market_df[market_df != 'KONEX']
    market_df = market_df[market_df != 0]

    # 3. market_type에 따라 필터링
    if market_type == None:
        filtered_df = market_df.dropna()
    else:
        filtered_df = market_df[market_df == market_type].dropna()

    return filtered_df.index.to_list()

def filterByHalt(halt_df, code_list:list):
    '''
    거래정지된 종목 제외
    :param halt_df: 거래정지 여부 데이터
    :param code_list: 종목코드
    :return:
    '''
    halt_df = halt_df[code_list]
    filtered_df = halt_df[halt_df == 0].dropna()

    return filtered_df.index.to_list()

def filterByMarketValue(mktval_df, code_list:list, top_n=200):
    '''
    시가총액 상위 N개 종목
    :param mktcap_df: 시가총액 데이터
    :param code_list: 종목코드
    :param top_n: 상위 N개 종목 int or list(top_n, bottom_n)
    :return:
    '''
    mktval_df = mktval_df[code_list]
    if type(top_n) == list:
        # if top_n is int, calculate top and bottom
        if type(top_n[0]) == int:
            top = top_n[0]
            bottom = top_n[1]
        # if top_n is float, calculate top and bottom
        elif type(top_n[0]) == float:
            top = int(top_n[0] * len(mktval_df))
            bottom = int(top_n[1] * len(mktval_df))
        else:
            raise ValueError('element of top_n(top, bottom) must be int or float')

        filtered_df = mktval_df[(mktval_df.rank(ascending=False) >= top)&(mktval_df.rank(ascending=False) <= bottom)].dropna()

    elif type(top_n) == int:
        if top_n == -1:
            filtered_df = mktval_df.dropna()
        else:
            filtered_df = mktval_df[mktval_df.rank(ascending=False) <= top_n].dropna()
    elif type(top_n) == float:
        top_n = int(top_n * len(mktval_df))
        filtered_df = mktval_df[mktval_df.rank(ascending=False) <= top_n].dropna()
    else:
        raise ValueError('top_n must be int or float or list')

    filtered_df = filtered_df.sort_values(ascending=False)

    return filtered_df.index.to_list()

def calculateMomentum(price_df, LT_window:int=52, ST_window:int=4):
    '''
    모멘텀 팩터(12m-1m) 계산
    :param price_df: 가격 데이터
    :param window: 모멘텀 팩터 계산 기간
    :return:
    '''
    momentum_df = price_df.pct_change(LT_window) - price_df.pct_change(ST_window)

    return momentum_df

def calculateLowVol(price_df, return_window:int=1, std_window:int=52):
    '''
    저변동성 팩터(52w) 계산
    :param price_df: 가격 데이터
    :param return_window: 가격 변동률 계산 기간
    :param std_window: 가격 변동률의 표준편차 계산 기간
    :return: lowVol_df
    '''

    # 작은 값이 좋도록 설정
    lowVol_df = -price_df.pct_change(return_window).rolling(std_window).std()

    return lowVol_df

def calcCoeff(data):
    y = data[0]
    x = data[1]
    window = data[2]

    rols = RollingOLS(y, x, window=window, missing='drop')
    rres = rols.fit()

    return pd.DataFrame(rres.params[x.columns[0]])

def calculateFSCoeff(price_bd, forsh_bd, window:int=52):
    '''
    외국인 지분 비중 계산
    :param price_df: 가격 데이터
    :param foreign_df: 외국인 지분 데이터
    :param window: 외국인 지분 비중 계산 기간
    :return:
    '''

    price_temp = price_bd[price_bd.index >= forsh_bd.index[0]]

    x_y_set = []
    for x in forsh_bd.columns:
        if forsh_bd[x].count() < window or price_temp[x].count() < window:
            continue
        x_y_set.append([price_temp[x], add_constant(forsh_bd[x], prepend=False), window])

    total_coeff = process_map(calcCoeff, x_y_set, max_workers=os.cpu_count()-1, desc=f'calc rolling regression', chunksize=10)
    total_coeff = pd.concat(total_coeff, axis=1)
    coeff = (price_bd - total_coeff.shift(1) * forsh_bd) / price_bd

    return coeff


def filterByFactor(factor_df, code_list:list, top_n=200):
    '''
    팩터가 가장 높은 상위 N개 종목
    :param factor_df: 팩터 데이터(반드시 높을수록 좋도록 설정)
    :param code_list: 종목코드
    :param top_n: 상위 N개 종목
    :return:
    '''
    factor_df = factor_df[code_list]

    if type(top_n) == list:
        # if top_n is int, calculate top and bottom
        if type(top_n[0]) == int:
            top = top_n[0]
            bottom = top_n[1]
        # if top_n is float, calculate top and bottom
        elif type(top_n[0]) == float:
            top = int(top_n[0] * len(factor_df))
            bottom = int(top_n[1] * len(factor_df))
        else:
            raise ValueError('element of top_n(top, bottom) must be int or float')

        filtered_df = factor_df[(factor_df.rank(ascending=False) >= top)&(factor_df.rank(ascending=False) <= bottom)].dropna()

    elif type(top_n) == int:
        if top_n == -1:
            filtered_df = factor_df.dropna()
        else:
            filtered_df = factor_df[factor_df.rank(ascending=False) <= top_n].dropna()
    elif type(top_n) == float:
        top_n = int(top_n * len(factor_df))
        filtered_df = factor_df[factor_df.rank(ascending=False) <= top_n].dropna()
    else:
        raise ValueError('top_n must be int or float or list')

    return filtered_df.index.to_list()

def weightEqual(code_list:list, mktval_df):
    '''
    동일가중 포트폴리오의 비중 계산
    :param code_list: 종목코드
    :param mktval_df: 시가총액 데이터
    :return:
    '''
    # 1. 동일가중 비중 계산
    weight = pd.Series(1/len(code_list), index=code_list, name=mktval_df.name)

    return weight

def weightMarktValue(code_list:list, mktval_df):
    '''
    시가총액 가중 포트폴리오의 비중 계산
    :param code_list: 종목코드
    :param mktval_df: 시가총액 데이터
    :return:
    '''
    # 1. 시가총액 비중 계산
    mktval_df = mktval_df[code_list]
    weight = mktval_df/mktval_df.sum()

    return weight

def weightFactor(code_list:list, factor_df):
    '''
    팩터 가중 포트폴리오의 비중 계산
    :param code_list: 종목코드
    :param factor_df: 팩터 데이터
    :return:
    '''
    # 1. 팩터 비중 계산
    factor_df = factor_df[code_list]
    # log factor_df
    log_factor = (1+factor_df).clip(upper=10)
    weight = log_factor/log_factor.sum()

    return weight

def limitWeight(weight, limit:float=0.3):
    '''
    비중값이 limit을 초과하는 경우 limit으로 설정하고, 나머지 종목들을 남은 비율에 맞추어 조정
    :param weight: 비중값 데이터프레임
    :param limit: 비중값 제한
    :return: 비중값이 제한된 비중값 데이터프레임
    '''
    # 1. 비중값이 limit을 초과하는 경우 limit으로 설정하고, 나머지 종목들을 남은 비율에 맞추어 조정
    # 제일 높은 값들을 하나하나 pop해서 limit을 초과하지 않게 조정해주는 식으로 처리하는게 제일 좋을 듯
    result = []
    total_val = 1
    while weight.max(axis=1).values[0] > limit and len(weight) > 0:
        target = weight.columns[np.argmax(weight)]
        overweight = pd.DataFrame(weight.pop(target))
        overweight[target] = limit
        result.append(overweight)
        total_val -= limit

        weight = weight * total_val / weight.sum(axis=1).values[0]

    result.append(weight)
    weight_df = pd.concat(result, axis=1)

    return weight_df