import numpy as np


def momentum(ts, lt, st):
    """
    Momentum factor
    :param ts_l: long term time series
    :param ts_s: short term time series
    :return: momentum factor
    """
    ts = (1+ts).cumprod().pct_change(lt) - (1+ts).cumprod().pct_change(st)

    return ts

def investment(ts):
    '''
    Investment 계산
    Inv = 당년 회계말 자산총계 / 전년 회계말 자산총계− 1
    :param ta: 자산총계 데이터
    :return: Inv
    '''
    #1년 전 데이터로 나누기
    ts_mask = ts.mask(ts <= 0, np.nan)
    inv = ts_mask / ts_mask.shift(52) - 1

    return inv

def value(ts):
    '''
    Value 계산
    :param ts: B/P 데이터 (build_data가 'bp'를 넘긴다 — 이미 book/price이므로 뒤집지 않는다)
    :return: B/P (클수록 value → longshort 상위 = long value)
    '''
    return ts

def size(ts):
    '''
    Size 계산
    :param mktcap: 시가총액 데이터
    :return: MktCap
    '''
    return 1 / ts

def profitability(ts):
    '''
    Profitability 계산
    :param roe: ROE 데이터
    :return: ROE
    '''
    return ts

def rev(ts, term):
    '''
    Revenue 계산
    :param rev: 매출액 데이터
    :return: Rev
    '''
    ts = - (1+ts).cumprod().pct_change(term)

    return ts