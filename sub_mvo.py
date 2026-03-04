import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize
import ffn
import pypfopt
from pypfopt import expected_returns, risk_models


class MeanVarianceOptimizer:
    def __init__(self, er, p):
        '''
        Mean Variance Optimizer
        1. Calculate sample covariance matrix
        2. Optimize weights to maximize Sharpe ratio
        3. Return optimized weights
        4. Can use shrinkage method for covariance matrix
        5. Can set bounds for weights
        :param er: er (expected return) snapshot dataframe
        :param p: p (price) dataframe
        '''
        self.er = er
        self.p = p
        self.r = p.pct_change().dropna(how='all')
        self.p_m = self.p.reindex(self.er.index)
        self.r_m = self.p_m.pct_change().dropna(how='all')

    @staticmethod
    def optimize(er, cov, bound: list = [0, 1], init_w=None, reg_lambda=None):
        n = len(er)
        if init_w is None:
            init_w = np.ones(n) / n
        bounds = [(bound[0], bound[1]) for _ in range(n)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        def neg_sharpe(w, mu, S):
            port_return = np.sum(w * mu)
            port_vol = np.sqrt(np.dot(w.T, np.dot(S, w)))

            sharpe_ratio = port_return / port_vol if port_vol != 0 else 0
            return -sharpe_ratio

        def loss_with_weight(w, w_init, mu, S, reg_lambda):
            loss = neg_sharpe(w, mu, S)
            weight_diff = np.sum((w - w_init) ** 2)
            return loss + reg_lambda * weight_diff

        if reg_lambda is None:
            result = minimize(neg_sharpe, init_w, args=(er, cov), method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            result = minimize(loss_with_weight, init_w, args=(init_w, er, cov, reg_lambda), method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x if result.success else None

    def calc_w(self, bound: list = [0, 1], shrink: bool = False, init_w=None, reg_lambda=None):
        min_idx = max(self.er.first_valid_index(), self.r.first_valid_index())
        er = self.er.loc[min_idx:].copy()
        r = self.r.loc[min_idx:].copy()
        total_w = []
        for date in er.index:
            curr_er = er.loc[date].dropna()
            if len(curr_er) < 2:
                continue
            if shrink:
                curr_p = self.p.loc[date - pd.DateOffset(weeks=52):date].copy()
                curr_p = curr_p[curr_er.index]
                curr_cov = risk_models.CovarianceShrinkage(curr_p, frequency=52).ledoit_wolf()
            else:
                curr_r = r.loc[date - pd.DateOffset(weeks=52):date]
                curr_r = curr_r[curr_er.index]
                curr_cov = curr_r.cov() * 52
            # set initial weights
            if init_w is not None:
                curr_w = init_w[curr_er.index].loc[curr_er.name].copy()
            else:
                curr_w = None
            w = self.optimize(curr_er, curr_cov, bound, curr_w, reg_lambda)
            total_w.append(pd.Series(w, index=curr_er.index, name=date))

        total_w_df = pd.DataFrame(total_w).sort_index()

        return total_w_df

    def backtest(self, w):
        # calc portfolio return based on total_w_df weights
        port_r_series = []
        for date in tqdm(w.index[1:]):
            prev_date = w.index[w.index.get_loc(date) - 1]
            curr_w = w.loc[prev_date].dropna()
            curr_ret = self.r_m.loc[date].copy()
            port_r = (curr_w * curr_ret[curr_w.index]).sum()
            port_r_series.append([date, port_r])
        port_r_df = pd.DataFrame(port_r_series, columns=['date', 'ER']).set_index('date')
        port_i_df = (1 + port_r_df).cumprod() * 100

        return port_i_df

    def display_result(self, port_i, bm, bm_label):
        # compare with equal weight
        tmp = pd.concat([port_i, *bm], axis=1)
        tmp.columns = ['MVO', *bm_label]
        ffn.calc_stats(tmp).display()

        return True

if __name__ == '__main__':
    country_list = [['us', 'RAY', 'ch4', 3, 12]
        , ['japan', 'TPX500', 'capm', 3, 60]
        , ['europe', 'SXXP', 'capm', 6, 12]
        , ['korea', 'KRX300', 'sr2', 3, 60]
        , ['china', 'MXCN', 'ff3', 12, 60]
    ]

    # step1: Get Weight for country using Expected Return of each country
    # read expected return and price data
    total_er = []
    total_prices = []
    total_vol = []
    for country in tqdm(country_list):
        country_name, index_name, model, avg, coeff = country
        data = pd.read_csv(f'./output/{country_name}/Expected_Return/{model}_{avg}m_{coeff}m_indexER.csv', index_col=0, parse_dates=True)
        er = data[['ER']].copy()
        mkt = data[['MKT']].copy()
        prices = pd.read_csv(f'./output/{country_name}/bm/BM.csv', index_col=0, parse_dates=True)
        vol = prices.pct_change().rolling(52).std() * np.sqrt(52)
        er.columns = [f'{country_name}']
        prices.columns = [f'{country_name}']
        vol.columns = [f'{country_name}']

        total_er.append(er)
        total_prices.append(prices)
        total_vol.append(vol)

    total_er = pd.concat(total_er, axis=1).sort_index()
    total_prices = pd.concat(total_prices, axis=1).sort_index()
    # mean variance optimization with er(expected return) and vol(standard deviation)
    # maximize return for sharpe ratio
    mvo = MeanVarianceOptimizer(total_er, total_prices)
    port_p = None
    w = mvo.calc_w(shrink=False, bound=[0, 1], reg_lambda=0.5).ffill().dropna(how='all')  # still 0.6, 0.5 best
    port_p = mvo.backtest(w)

    # compare with equal weight
    # calc equal port
    ew_p = mvo.r_m.mean(axis=1).copy()
    ew_p = (1 + ew_p).cumprod() * 100
    _ = mvo.display_result(port_p, [ew_p], ['Equal Weight'])

    country_w = w.copy()

    # step2: Get Weight for sector using Expected Return of each sector in each country

    sector_list = ['10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']

    total_port = []
    # read expected return and price data
    for country in tqdm(country_list):
        print('Processing: ', country)
        total_er = []
        total_prices = []
        total_vol = []
        total_mkt = []
        country_name, index_name, model, avg, coeff = country
        for sector in sector_list:
            data = pd.read_csv(f'./output/{country_name}/Expected_Return/{model}_{avg}m_{coeff}m_{sector}ER.csv', index_col=0, parse_dates=True)
            er = data[['ER']].copy()
            mkt = data[['MKT']].copy()
            prices = pd.read_csv(f'./output/{country_name}/bm/BM{sector}.csv', index_col=0, parse_dates=True)
            vol = prices.pct_change().rolling(52).std() * np.sqrt(52)
            er.columns = [f'{sector}']
            prices.columns = [f'{sector}']
            vol.columns = [f'{sector}']
            mkt.columns = [f'{sector}']

            total_er.append(er)
            total_prices.append(prices)
            total_vol.append(vol)
            total_mkt.append(mkt)

        total_er = pd.concat(total_er, axis=1).sort_index()
        total_prices = pd.concat(total_prices, axis=1).sort_index()
        total_mkt = pd.concat(total_mkt, axis=1).sort_index()
        init_w = total_mkt.div(total_mkt.sum(axis=1), axis=0)

        # mean variance optimization with er(expected return) and vol(standard deviation)
        # maximize return for sharpe ratio
        mvo = MeanVarianceOptimizer(total_er, total_prices)
        port_p = None
        w = mvo.calc_w(shrink=False, bound=[0,1], init_w=init_w, reg_lambda=0.5).ffill().dropna(how='all') # still 0.6, 0.5 best
        port_p = mvo.backtest(w)

        # compare with equal weight
        # load bm
        bm = pd.read_csv(f'./output/{country_name}/bm/BM.csv', dtype=float, index_col=0, parse_dates=True)
        bm = bm.reindex(port_p.index).ffill().dropna(how='all')
        # calc equal port
        ew_p = mvo.r_m.mean(axis=1).copy()
        ew_p = (1 + ew_p).cumprod() * 100
        _ = mvo.display_result(port_p, [bm, ew_p], ['Market Weight', 'Equal Weight'])

        port_p.columns = [f'{country_name}']
        total_port.append(port_p)

    total_port = pd.concat(total_port, axis=1).sort_index()

    # get sector optimized country portfolio with weight calc from step1
    total_port_r = total_port.pct_change().dropna(how='all')
    w_r = total_port_r.mul(country_w.shift(1))

    combined_port = (1+w_r.sum(axis=1)).cumprod()

    # compare with equal weight
    # calc equal port
    combined_port = combined_port['2003-04-25':]
    ffn.calc_stats(combined_port).display()