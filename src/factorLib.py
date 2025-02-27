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