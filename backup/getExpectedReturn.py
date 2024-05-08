from src.funs import *
from src.calcEReturn import Calculator

def get_eval_df(filepath, window, reverse=False):
    c = load_pickle(filepath)

    if reverse:
        # change sign of c
        newc = []
        for eachc in c:
            eachc = eachc.apply(lambda x: x * 0.693 if x > 0 else x / 0.693)
            newc.append(-eachc)
        c = newc

    p = cal.evaluate(c, window)

    return p.dropna()

def get_eval_df_sector(filepath, sectordf, sector, window, reverse=False):
    c = load_pickle(filepath)

    # change sign of c
    newc = []
    for eachc in c:
        sectorc = sectordf[eachc.index][sectordf.index<=eachc.name].iloc[-1].str[:len(sector)]
        filteredc = eachc[sectorc == sector]
        if reverse:
            newc.append(-filteredc)
        else:
            newc.append(filteredc)
    c = newc

    p = cal.evaluate(c, window)

    return p.dropna()

if __name__ == '__main__':
    #for country in ['us', 'korea', 'japan', 'europe']:
    country = 'korea'
    reverse = True
    print(f'country: {country}, mean reverting: {reverse}')

    rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
    price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    halt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    mkt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
    ind_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)
    # ind_df의 기간을 price_bd에 맞추기
    ind_df = ind_df.reindex(price_df.index).ffill().bfill().dropna(axis=0, how='all')
    bv_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_bookvalue.csv', dtype=float, index_col=0, parse_dates=True)
    bm_df = pd.read_csv(f'./output/{country}/bm/price_bm_m.csv', dtype=float, index_col=0, parse_dates=True)
    smb_df = pd.read_csv(f'./output/{country}/bm/price_smb_m.csv', dtype=float, index_col=0, parse_dates=True)
    hml_df = pd.read_csv(f'./output/{country}/bm/price_hml_m.csv', dtype=float, index_col=0, parse_dates=True)
    mom_df = pd.read_csv(f'./output/{country}/bm/price_mom_m.csv', dtype=float, index_col=0, parse_dates=True)
    op_df = pd.read_csv(f'./output/{country}/bm/price_op_m.csv', dtype=float, index_col=0, parse_dates=True)
    inv_df = pd.read_csv(f'./output/{country}/bm/price_inv_m.csv', dtype=float, index_col=0, parse_dates=True)

    # Unify index of all data to max lower bound
    basedate = findBaseDate(rf_df, price_df, halt_df, mkt_df, bm_df, smb_df, hml_df, mom_df, op_df, inv_df, ind_df)
    rf_df = rf_df[rf_df.index >= basedate]
    price_df = price_df[price_df.index >= basedate]
    halt_df = halt_df[halt_df.index >= basedate]
    mkt_df = mkt_df[mkt_df.index >= basedate]
    ind_df = ind_df[ind_df.index >= basedate]
    bm_df = bm_df[bm_df.index >= basedate]
    # also unify index for factors
    smb_df = smb_df[smb_df.index >= basedate]
    hml_df = hml_df[hml_df.index >= basedate]
    mom_df = mom_df[mom_df.index >= basedate]
    op_df = op_df[op_df.index >= basedate]
    inv_df = inv_df[inv_df.index >= basedate]

    cal = Calculator(price=price_df, halt=halt_df, rf=rf_df, bm=bm_df)
    #datawindow = 12  # 사용할 데이터의 window
    #basewindow = 14  # 넉넉하게 부여
    #c = cal.run(method='ff6', resampl='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=3)
    #save_pickle(c, f'./output/{country}/Expected_Return/ff6_3m_12m.pkl')


    '''
    datawindow = 12  # 사용할 데이터의 window
    basewindow = 14  # 넉넉하게 부여
    
    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=1)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_1m_12m.pkl')
    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_3m_12m.pkl')

    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=1)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_1m_12m.pkl')
    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_3m_12m.pkl')

    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                avg_window=1)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_1m_12m.pkl')
    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_3m_12m.pkl')

    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                mom=mom_df, avg_window=1)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_1m_12m.pkl')
    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                mom=mom_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_3m_12m.pkl')

    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df,
                inv=inv_df, avg_window=1)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_1m_12m.pkl')
    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df,
                inv=inv_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_3m_12m.pkl')

    datawindow = 36  # 사용할 데이터의 window
    basewindow = 38  # 넉넉하게 부여

    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_3m_36m.pkl')
    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_12m_36m.pkl')

    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_3m_36m.pkl')
    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_12m_36m.pkl')

    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_3m_36m.pkl')
    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_12m_36m.pkl')

    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                mom=mom_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_3m_36m.pkl')
    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df,
                mom=mom_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_12m_36m.pkl')

    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df,
                inv=inv_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_3m_36m.pkl')
    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df,
                inv=inv_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_12m_36m.pkl')

    datawindow = 60 # 사용할 데이터의 window
    basewindow = 62 # 넉넉하게 부여

    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_3m_60m.pkl')
    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=datawindow, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_12m_60m.pkl')

    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_3m_60m.pkl')
    c = cal.run(method='capm', resample='M', basewindow=basewindow, window=datawindow, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/capm_12m_60m.pkl')

    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_3m_60m.pkl')
    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_12m_60m.pkl')

    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_3m_60m.pkl')
    c = cal.run(method='ch4', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ch4_12m_60m.pkl')

    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_3m_60m.pkl')
    c = cal.run(method='ff5', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_12m_60m.pkl')'''

    '''
    window = 1

    # set mapped index of industry
    if country == 'korea':
        sector_class_len = 3
    else:
        sector_class_len = 2

    map_ind = pd.read_csv(f'D:/Factor_DB/{country}/{country}_industrymap.csv', dtype=str, index_col=0)
    map_ind.index = map_ind.index.map(str)
    unique_ind = ind_df.iloc[-1].str[:sector_class_len].drop_duplicates().dropna().unique()
    ind_list = map_ind.loc[unique_ind]

    total_result = []
    for ind in ind_list.index:
        pred = get_eval_df_sector(f'./output/{country}/Expected_Return/ff5_12m_36m.pkl', ind_df, ind, window, reverse=reverse)
        result = pred.mean()
        result.name = ind
        total_result.append(result)
    pd.DataFrame(total_result).to_csv(f'./{country}ff5_12m_36m_sector.csv')
    '''

    window = 1

    #mean_3m_12m = get_eval_df(f'./output/{country}/Expected_Return/mean_3m_12m.pkl', window)
    capm_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/capm_12m_60m.pkl', window)
    #ff3_3m_12m = get_eval_df(f'./output/{country}/Expected_Return/ff3_3m_12m.pkl', window)
    #ch4_3m_12m = get_eval_df(f'./output/{country}/Expected_Return/ch4_3m_12m.pkl', window)
    #ff5_3m_12m = get_eval_df(f'./output/{country}/Expected_Return/ff5_3m_12m.pkl', window)
    #ff6_3m_12m = get_eval_df(f'./output/{country}/Expected_Return/ff6_3m_12m.pkl', window)

    #print('mean_3m_12m : \n', mean_3m_12m.median())
    print('capm_3m_12m : \n', capm_12m_60m.median())
    #print('ff3_3m_12m : \n', ff3_3m_12m.median())
    #print('ch4_3m_12m : \n', ch4_3m_12m.median())
    #print('ff5_3m_12m : \n', ff5_3m_12m.median())
    #print('ff6_3m_12m : \n', ff6_3m_12m.median())

    print('done')


    '''Note
    - 보다 장기 window의 regression 베타와 장기 mean vector를 사용하면 할수록 CAPM이 단순 과거 수익률 평균보다 낮은 MSFE와 높은 IR을 기록
    - FF3, CH4 모델은 CAPM보다 높은 MSFE와 낮은 IR을 기록
    - FF3, CH4 모델은 장기 window의 regression 베타와 장기 mean vector를 사용하면 할수록 MSFE가 높아지고 IR이 낮아짐
    - 다만, IC와의 관계는 불확실, 한국은 (-)의 관계가 나오지만, 미국은 (+)의 관계가 나옴
    - 한국은 FF3 < CH4 관계가 성립되나, 미국은 다소 불확실함(MSFE 기준)
    #하 시발 어렵다'''

