def momentum(rtn, ts_l, ts_s):
    """
    Momentum factor
    :param ts_l: long term time series
    :param ts_s: short term time series
    :return: momentum factor
    """
    ts = rtn.prod(LT_window) - price_df.pct_change(ST_window)

    return ts