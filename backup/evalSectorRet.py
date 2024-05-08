from glob import glob

from src.funs import *
from src.calcEReturn import Calculator

def get_eval_df(filepath, window):
    c = load_pickle(filepath)

    p = cal.evaluate(c, window)

    return p.dropna()


if __name__ == '__main__':
    for country in ['korea']:

        rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
        price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
        mkt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
        industry_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_industry.csv', dtype=str, index_col=0, parse_dates=True)
        industry_df = industry_df.reindex(index=price_df.index).ffill().dropna(how='all', axis=0).bfill()

        # Unify index of all data to max lower bound
        basedate = findBaseDate(rf_df, price_df, mkt_df)
        rf_df = rf_df[rf_df.index >= basedate]
        price_df = price_df[price_df.index >= basedate]
        mkt_df = mkt_df[mkt_df.index >= basedate]

        window = 1

        sec_bm_list = glob(f'./output/{country}/industry_return/sector*.csv')
        method_list = glob(f'./output/{country}/Expected_Return/*.pkl')
        method_list = glob(f'./output/{country}/Expected_Return/capm_12m_60m.pkl')

        result = []
        for method in method_list:
            for sec_bm in sec_bm_list:
                sec_bm_df = pd.read_csv(sec_bm, index_col=0, parse_dates=True)
                sec_bm_df = sec_bm_df[sec_bm_df.index >= basedate]
                sec_bm_rtn = sec_bm_df.resample('M').last().pct_change().dropna(axis=1, how='all')
                rf = rf_df.resample('M').last() / 100
                rf = (rf + 1) ** (1/12) - 1

                sec_code = sec_bm.split('\\')[-1].split('+')[0].replace('sector', '')

                # calc expected return of sector
                expret = load_pickle(method)
                d = []
                for er in expret:
                    #retrieve industry info
                    currdate = er.name
                    industry = industry_df[industry_df.index <= currdate].iloc[-1].str[:len(sec_code)]
                    industry = industry[industry == sec_code]
                    # remain only if er index and industry_df index are same
                    er = er[er.index.isin(industry.index)]
                    mkt = mkt_df[mkt_df.index <= currdate].iloc[-1][er.index]
                    # weighted average of er and mkt_df
                    er = (er * mkt).sum() / mkt.sum()
                    er = pd.Series(er, index=sec_bm_df.columns, name=currdate)
                    d.append(er)

                cal = Calculator(price=sec_bm_df, rf=rf_df)
                p = cal.evaluate(d, window)

                # calc information coefficient
                d_df = pd.DataFrame(d)
                # shift sec_bm_rtn and align date with d_df
                sec_bm_rtn_clip = sec_bm_rtn.shift(-1)[sec_bm_rtn.index.isin(d_df.index)]
                # shift rf and align date with d_df
                rf_clip = rf.shift(-1)[rf.index.isin(d_df.index)]
                # calc excess return
                sec_bm_rtn_clip = sec_bm_rtn_clip - rf_clip.values
                # calc correlation btw act and exp return
                ic = np.corrcoef(d_df[d_df.columns[0]].values, sec_bm_rtn_clip[sec_bm_rtn_clip.columns[0]].values)

                result.append([method, sec_code, p.mean(axis=0).MSFE, ic[0][1]])

        result_df = pd.DataFrame(result, columns=['method', 'sector', 'MSFE', 'IC'])
        result_df.to_csv(f'{country}result.csv')
    print('done')
