import itertools

from src.funs import *
from src.calcEReturn import Calculator

def get_eval_df(filepath, window):
    c = load_pickle(filepath)

    p = cal.evaluate(c, window)

    return p.dropna()

def get_eval_df_sector(filepath, sectordf, sector, window):
    c = load_pickle(filepath)

    # change sign of c
    newc = []
    for eachc in c:
        sectorc = sectordf[eachc.index][sectordf.index<=eachc.name].iloc[-1].str[:len(sector)]
        filteredc = eachc[sectorc == sector]
        newc.append(filteredc)

    c = newc

    p = cal.evaluate(c, window)

    return p.dropna()

def get_eval_df_size(filepath, mktdf, size, window):
    c = load_pickle(filepath)

    newc = []
    for eachc in c:
        sizec = pd.qcut(mktdf[eachc.index][mktdf.index<=eachc.name].iloc[-1], 3, labels=['small', 'mid', 'large'])
        filteredc = eachc[sizec == size]
        newc.append(filteredc)

    c = newc

    p = cal.evaluate(c, window)

    return p.dropna()

if __name__ == '__main__':
    #country_list = ['europe'] # ['us', 'korea', 'japan', 'europe']
    #avgwindow_list = [6]
    #datawindow_list = [12]
    #iterlist = list(itertools.product(country_list, avgwindow_list, datawindow_list))

    iterlist = [
        ['us',3,12],['europe',6,12], ['korea', 1, 60],['japan', 6, 60]
    ]#,['us',3,12],['europe',6,12]] #['korea', 1, 60],['japan', 6, 60]

    #iterlist_sector = [
    #    ['korea', 'capm', True, 1, 36, 3]
    #    , ['us', 'ch4', False, 3, 12, 2]
    #    , ['europe', 'capm', False, 6, 12, 2]
    #    , ['japan', 'capm', True, 6, 60, 2]]
    #iterlist = [['korea', False, 1, 60]]

    for country, avgwindow, datawindow in iterlist:
        if avgwindow >= datawindow:
            continue
        #country = 'us'
        print(f'country: {country}, avgwindow: {avgwindow} datawindow: {datawindow}')

        rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
        price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
        halt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
        mkt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
        ind_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)
        # ind_df의 기간을 price_bd에 맞추기
        ind_df = ind_df.reindex(price_df.index).ffill().bfill().dropna(axis=0, how='all')
        bm_df = pd.read_csv(f'./output/{country}/bm/BM_m.csv', dtype=float, index_col=0, parse_dates=True)
        smb_df = pd.read_csv(f'./output/{country}/bm/SMB_m.csv', dtype=float, index_col=0, parse_dates=True)
        hml_df = pd.read_csv(f'./output/{country}/bm/HML_m.csv', dtype=float, index_col=0, parse_dates=True)
        mom_df = pd.read_csv(f'./output/{country}/bm/MOM_m.csv', dtype=float, index_col=0, parse_dates=True)
        op_df = pd.read_csv(f'./output/{country}/bm/OP_m.csv', dtype=float, index_col=0, parse_dates=True)
        inv_df = pd.read_csv(f'./output/{country}/bm/INV_m.csv', dtype=float, index_col=0, parse_dates=True)

        # cai 불러오기
        cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
        cai_bd = cai_bd.reindex(price_df.index, method='bfill').ffill()
        cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
        cai_bd = cai_bd.fillna(1)

        # cai를 기준으로 share관련 지표들 수정
        price_df *= cai_bd

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

        basedate = findBaseDate(rf_df, price_df, halt_df, mkt_df, bm_df)
        rf_df = rf_df[rf_df.index >= basedate]
        price_df = price_df[price_df.index >= basedate]
        halt_df = halt_df[halt_df.index >= basedate]
        mkt_df = mkt_df[mkt_df.index >= basedate]
        bm_df = bm_df[bm_df.index >= basedate]

        cal = Calculator(price=price_df, halt=halt_df, rf=rf_df, bm=bm_df)

        c = cal.run(method='sr2', resample='M', window=datawindow, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/sr2_{avgwindow}m_{datawindow}m.pkl')

        c = cal.run(method='mean', resample='M', window=datawindow, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/mean_{avgwindow}m_{datawindow}m.pkl')

        c = cal.run(method='capm', resample='M', window=datawindow, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/capm_{avgwindow}m_{datawindow}m.pkl')

        c = cal.run(method='ff3', resample='M', window=datawindow, hml=hml_df, smb=smb_df, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/ff3_{avgwindow}m_{datawindow}m.pkl')

        c = cal.run(method='ff5', resample='M', window=datawindow, hml=hml_df, smb=smb_df, op=op_df, inv=inv_df, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/ff5_{avgwindow}m_{datawindow}m.pkl')

        c = cal.run(method='ch4', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, avg_window=avgwindow)
        save_pickle(c, f'./output/{country}/Expected_Return/ch4_{avgwindow}m_{datawindow}m.pkl')


        #sr2 = get_eval_df(f'./output/{country}/Expected_Return/sr2_{avgwindow}m_{datawindow}m.pkl', 1)
        #print(f'sr2 : ', pd.concat([sr2.mean(), sr2.median()], axis=1))

        '''
        if os.path.isfile(f'./output/{country}/Expected_Return/capm_{avgwindow}m_{datawindow}m.pkl') == False:
            c = cal.run(method='capm', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=avgwindow)
            save_pickle(c, f'./output/{country}/Expected_Return/capm_{avgwindow}m_{datawindow}m.pkl')
        if os.path.isfile(f'./output/{country}/Expected_Return/ff3_{avgwindow}m_{datawindow}m.pkl') == False:
            c = cal.run(method='ff3', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=avgwindow)
            save_pickle(c, f'./output/{country}/Expected_Return/ff3_{avgwindow}m_{datawindow}m.pkl')
        if os.path.isfile(f'./output/{country}/Expected_Return/ff5_{avgwindow}m_{datawindow}m.pkl') == False:
            c = cal.run(method='ff5', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=avgwindow)
            save_pickle(c, f'./output/{country}/Expected_Return/ff5_{avgwindow}m_{datawindow}m.pkl')
        if os.path.isfile(f'./output/{country}/Expected_Return/ch4_{avgwindow}m_{datawindow}m.pkl') == False:
            c = cal.run(method='ch4', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=avgwindow)
            save_pickle(c, f'./output/{country}/Expected_Return/ch4_{avgwindow}m_{datawindow}m.pkl')
        if os.path.isfile(f'./output/{country}/Expected_Return/mean_{avgwindow}m_{datawindow}m.pkl') == False:
            c = cal.run(method='mean', resample='M', window=datawindow, hml=hml_df, smb=smb_df, mom=mom_df, op=op_df, inv=inv_df, avg_window=avgwindow)
            save_pickle(c, f'./output/{country}/Expected_Return/mean_{avgwindow}m_{datawindow}m.pkl')
        '''
        '''
        mean = get_eval_df(f'./output/{country}/Expected_Return/mean_{avgwindow}m_{datawindow}m.pkl', 1, reverse)
        capm = get_eval_df(f'./output/{country}/Expected_Return/capm_{avgwindow}m_{datawindow}m.pkl', 1, reverse)
        ff3 = get_eval_df(f'./output/{country}/Expected_Return/ff3_{avgwindow}m_{datawindow}m.pkl', 1, reverse)
        ch4 = get_eval_df(f'./output/{country}/Expected_Return/ch4_{avgwindow}m_{datawindow}m.pkl', 1, reverse)
        ff5 = get_eval_df(f'./output/{country}/Expected_Return/ff5_{avgwindow}m_{datawindow}m.pkl', 1, reverse)

        print(f'{avgwindow}m_{datawindow}m Test Result: Mean/ Median')
        print(f'mean : ', pd.concat([mean.mean(), mean.median()], axis=1))
        print(f'capm : ', pd.concat([capm.mean(), capm.median()], axis=1))
        print(f'ff3 : ', pd.concat([ff3.mean(), ff3.median()], axis=1))
        print(f'ch4 : ', pd.concat([ch4.mean(), ch4.median()], axis=1))
        print(f'ff5 : ', pd.concat([ff5.mean(), ff5.median()], axis=1))
        '''

        '''
        model = 'sr2'
        
        if country == 'korea':
            sectorlen = 3
        else:
            sectorlen = 2

        sectors = ind_df.iloc[-1]
        sectors = pd.DataFrame(set(sectors.str[:sectorlen])).dropna()

        result_list = []
        for sec in sectors[0].values:
            result = get_eval_df_sector(f'./output/{country}/Expected_Return/{model}_{avgwindow}m_{datawindow}m.pkl', ind_df, sec, 1)
            result_df = pd.concat([result.mean(), result.median()], axis=0)
            result_df.name = sec
            result_list.append(result_df)
            print(f'{country} {sec} : ', pd.concat([result.mean(), result.median()], axis=1))

        result_df = pd.concat(result_list, axis=1)
        result_df.to_csv(f'./{country}_{model}_{avgwindow}m_{datawindow}m_sector.csv')'''


        '''
        model = 'sr2'
        sizes = ['small', 'mid', 'large']
        result_list = []
        for size in sizes:
            result = get_eval_df_size(f'./output/{country}/Expected_Return/{model}_{avgwindow}m_{datawindow}m.pkl', mkt_df, size, 1)
            result_df = pd.concat([result.mean(), result.median()], axis=0)
            result_df.name = size
            result_list.append(result_df)
            print(f'{country} {size} : ', pd.concat([result.mean(), result.median()], axis=1))

        result_df = pd.concat(result_list, axis=1)
        result_df.to_csv(f'./{country}_{model}_{avgwindow}m_{datawindow}m_size.csv')
        print('done')'''

