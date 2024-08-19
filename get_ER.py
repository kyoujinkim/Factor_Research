from src.funs import *
from src.calcEReturn import Calculator

def calculate_Expected_Return(country, model, avg_window, window, end: str = None):
    '''
    기대수익률을 계산

    :param country: 국가명
    :param model: 모델명 abbreviation (e.g. CAPM, ff5 ...)
    :param avg_window: 미래 팩터 수익률로 사용하기 위한 평균값 window
    :param window: 회귀계수를 계산하기 위한 데이터 window
    :param end: 기대수익률을 계산할 최종날짜
    :return:
    '''

    basewindow = window+2  # 넉넉하게 부여
    print(f'country: {country}')

    rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
    price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    halt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    ind_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)
    # ind_df의 기간을 price_bd에 맞추기
    ind_df = ind_df.reindex(price_df.index).ffill().bfill().dropna(axis=0, how='all')

    # cai 불러오기
    cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
    cai_bd = cai_bd.reindex(price_df.index, method='bfill').ffill()
    cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
    cai_bd = cai_bd.fillna(1)

    # cai를 기준으로 share관련 지표들 수정
    price_df = price_df * cai_bd

    # load factor data
    bm_df = pd.read_csv(f'./output/{country}/bm/BM_m.csv', dtype=float, index_col=0, parse_dates=True)
    smb_df = pd.read_csv(f'./output/{country}/bm/SMB_m.csv', dtype=float, index_col=0, parse_dates=True)
    hml_df = pd.read_csv(f'./output/{country}/bm/HML_m.csv', dtype=float, index_col=0, parse_dates=True)
    mom_df = pd.read_csv(f'./output/{country}/bm/MOM_m.csv', dtype=float, index_col=0, parse_dates=True)
    op_df = pd.read_csv(f'./output/{country}/bm/OP_m.csv', dtype=float, index_col=0, parse_dates=True)
    inv_df = pd.read_csv(f'./output/{country}/bm/INV_m.csv', dtype=float, index_col=0, parse_dates=True)

    # Unify index of all data to max lower bound
    basedate = findBaseDate(rf_df, price_df, halt_df, bm_df, smb_df, hml_df, mom_df, op_df, inv_df, ind_df)
    rf_df = rf_df[rf_df.index >= basedate]
    price_df = price_df[price_df.index >= basedate]
    halt_df = halt_df[halt_df.index >= basedate]
    bm_df = bm_df[bm_df.index >= basedate]
    # also unify index for factors
    smb_df = smb_df[smb_df.index >= basedate]
    hml_df = hml_df[hml_df.index >= basedate]
    mom_df = mom_df[mom_df.index >= basedate]
    op_df = op_df[op_df.index >= basedate]
    inv_df = inv_df[inv_df.index >= basedate]

    cal = Calculator(price=price_df, halt=halt_df, rf=rf_df, bm=bm_df)

    c = cal.run_snapshot(method=model, resample='M', basewindow=basewindow, window=window, mom= mom_df, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=avg_window, end=end)
    betas = cal.run_snapshot(method='ff6', resample='M', basewindow=basewindow, window=window, mom= mom_df, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=avg_window, end=end, view_betas=True)

    c.to_csv(f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m_snapshot{c.name.strftime("%Y-%m-%d")}.csv')
    betas.to_csv(f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m_snapshot_betas_{c.name.strftime("%Y-%m-%d")}.csv')

    return c.name.strftime("%Y-%m-%d")

def calculate_Expected_Return_totalTS(country, model, avg_window, window, end: str = None):
    '''
    기대수익률을 계산

    :param country: 국가명
    :param model: 모델명 abbreviation (e.g. CAPM, ff5 ...)
    :param avg_window: 미래 팩터 수익률로 사용하기 위한 평균값 window
    :param window: 회귀계수를 계산하기 위한 데이터 window
    :param end: 기대수익률을 계산할 최종날짜
    :return:
    '''

    basewindow = window+2  # 넉넉하게 부여
    print(f'country: {country}')

    rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
    price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    halt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    ind_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)
    # ind_df의 기간을 price_bd에 맞추기
    ind_df = ind_df.reindex(price_df.index).ffill().bfill().dropna(axis=0, how='all')

    # cai 불러오기
    cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
    cai_bd = cai_bd.reindex(price_df.index, method='bfill').ffill()
    cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
    cai_bd = cai_bd.fillna(1)

    # cai를 기준으로 share관련 지표들 수정
    price_df = price_df * cai_bd

    # load factor data
    bm_df = pd.read_csv(f'./output/{country}/bm/BM_m.csv', dtype=float, index_col=0, parse_dates=True)
    smb_df = pd.read_csv(f'./output/{country}/bm/SMB_m.csv', dtype=float, index_col=0, parse_dates=True)
    hml_df = pd.read_csv(f'./output/{country}/bm/HML_m.csv', dtype=float, index_col=0, parse_dates=True)
    mom_df = pd.read_csv(f'./output/{country}/bm/MOM_m.csv', dtype=float, index_col=0, parse_dates=True)
    op_df = pd.read_csv(f'./output/{country}/bm/OP_m.csv', dtype=float, index_col=0, parse_dates=True)
    inv_df = pd.read_csv(f'./output/{country}/bm/INV_m.csv', dtype=float, index_col=0, parse_dates=True)

    # Unify index of all data to max lower bound
    basedate = findBaseDate(rf_df, price_df, halt_df, bm_df, smb_df, hml_df, mom_df, op_df, inv_df, ind_df)
    rf_df = rf_df[rf_df.index >= basedate]
    price_df = price_df[price_df.index >= basedate]
    halt_df = halt_df[halt_df.index >= basedate]
    bm_df = bm_df[bm_df.index >= basedate]
    # also unify index for factors
    smb_df = smb_df[smb_df.index >= basedate]
    hml_df = hml_df[hml_df.index >= basedate]
    mom_df = mom_df[mom_df.index >= basedate]
    op_df = op_df[op_df.index >= basedate]
    inv_df = inv_df[inv_df.index >= basedate]

    cal = Calculator(price=price_df, halt=halt_df, rf=rf_df, bm=bm_df)

    c = cal.run(method=model, resample='M', basewindow=basewindow, window=window, mom= mom_df, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=avg_window, end=end)

    save_pickle(c, f'./output/{country}/Expected_Return/{model}_{avg_window}m_{window}m.pkl')

    return c.name.strftime("%Y-%m-%d")