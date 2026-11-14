import itertools
import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
warnings.simplefilter(action='ignore', category=FutureWarning)

# factor regressors of each model besides bm, in regression order (see Calculator._factor_model)
MODEL_FACTORS = {
    'capm': [],
    'ff3': ['hml', 'smb'],
    'ch4': ['hml', 'smb', 'mom'],
    'rv4': ['hml', 'smb', 'rev'],
    'ff5': ['hml', 'smb', 'op', 'inv'],
    'ff6': ['hml', 'smb', 'op', 'inv', 'mom'],
    'hs': ['hs'],
}
# factors used as given (not an index level turned into returns)
RAW_FACTORS = {'hs'}

class Calculator:
    '''
    계산기 클래스
    '''

    def __init__(self, index:pd.DataFrame, price:pd.DataFrame, rf: pd.DataFrame=None, bm: pd.DataFrame=None):
        self.index = index
        self.price_bd = price
        if rf is None:
            #set rf as 0 if not given
            self.rf_bd = pd.DataFrame(0, index=price.index)
        else:
            # since rf is given as percentage
            self.rf_bd = rf.ffill() / 100
        if bm is None:
            self.bm_bd = pd.DataFrame(np.ones(len(price.index)), index=price.index, columns=['bm'])
        else:
            self.bm_bd = bm.ffill()

    def __getstate__(self):
        # workers of run() only need the resampled frames; skip the full weekly history when pickling
        return {k: v for k, v in self.__dict__.items() if k not in ('price_bd', 'rf_bd', 'bm_bd')}

    def run(self, method:str, factors:Dict={}, resample:str='M', window:int=12, **kwargs) -> list:
        '''
        날짜, 예측값, 실제값의 리스트를 반환하는 함수
        :param method: 예측값을 계산하는 방법
        :param resample: 수익률 계산시에 사용할 리샘플링 날짜 단위
        :param basewindow: 수익률 계산에 사용할 기간(넉넉하게)
        :param window: 예측값 계산에 사용할 날짜 기간
        :param kwargs: 추가적인 인자
            smb: small minus large factor return index
            hml: High minus Low factor return index
            mom: Momentum factor return index
            op: Operating Profitability factor return index
            inv: Investment factor return index
            strev: Short Term Reversal factor return per equities
            avg_window: historical mean return when calculate exp_return
            reverse: True if you want negative result
        :return: 날짜, 예측값, 실제값의 리스트
        '''

        port_dates = pd.date_range(start=self.price_bd.index[0], end=self.price_bd.index[-1], freq='W-FRI')
        port_dates = pd.Series(port_dates, index=port_dates).resample(resample).last()
        self.price = self.price_bd.reindex(port_dates.values, method='ffill').dropna(how='all')
        self.rf = self.rf_bd.reindex(port_dates.values, method='ffill').dropna(how='all')
        self.bm = self.bm_bd.reindex(port_dates.values, method='ffill').dropna(how='all')
        basewindow = window + 2

        self.factors = factors.copy()
        for f in self.factors.keys():
            self.factors[f] = self.factors[f].reindex(port_dates.values, method='ffill').dropna(how='all')
        #assign kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.avg_window = kwargs.get('avg_window', 12)
        view_betas = kwargs.get('view_betas', False)

        if resample=='M':
            bday = 12
        elif resample=='W':
            bday = 52
        else:
            bday = 252

        iterator = list(itertools.product([method], [bday], [basewindow], [window], range(basewindow+1, len(self.price)+1), [view_betas]))
        workers = os.cpu_count()-2
        # every task pickles self, so hand each worker a batch of dates instead of one
        result = process_map(self._iterrun, iterator, max_workers=workers, chunksize=max(1, len(iterator) // (workers * 4)))

        return result

    def run_snapshot(self, method:str, factors:Dict={}, resample:str='M', basewindow:int=24, window:int=12, view_betas:bool=False, end:str=None, **kwargs) -> pd.DataFrame:
        '''
        지정된 모델에 대해 날짜별 예측값을 반환하는 함수

        :param method: method to calculate expected return
        :param resample: resample frequency
        :param basewindow: basewindow for return calculation
        :param window: window for expected return calculation
        :param kwargs: additional arguments
        :return: dataframe of expected return
        '''
        if type(end)==str:
            end = pd.to_datetime(end)
        end = min(end, self.price_bd.index[-1]) if end is not None else self.price_bd.index[-1]
        port_dates = pd.date_range(start=self.price_bd.index[0], end=end, freq='W-FRI')
        port_dates = pd.Series(port_dates, index=port_dates).resample(resample).last()
        self.price = self.price_bd[self.price_bd.index <= end].reindex(port_dates.values, method='ffill').dropna(how='all')
        self.rf = self.rf_bd.reindex(port_dates.values, method='ffill').dropna(how='all')
        self.bm = self.bm_bd.reindex(port_dates.values, method='ffill').dropna(how='all')

        self.factors = factors.copy()
        for f in self.factors.keys():
            self.factors[f] = self.factors[f].reindex(port_dates.values, method='ffill').dropna(how='all')
        #assign kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.avg_window = kwargs.get('avg_window', 12)

        if resample=='M':
            bday = 12
        elif resample=='W':
            bday = 52
        else:
            raise ValueError('Resample should be either M or W')

        result = self._iterrun([method, bday, basewindow, window, len(self.price), view_betas])

        return result

    @staticmethod
    def _check_factors(method, factor):
        if method == 'mean':
            return True
        if method == 'sr2':
            return factor == 'bm'
        return method in MODEL_FACTORS and factor in ['bm'] + MODEL_FACTORS[method]

    def _iterrun(self, var):

        method, bday, basewindow, window, i, view_betas = var

        temp_price = self.price.iloc[i - (basewindow + 1):i]
        temp_date = temp_price.index[-1]

        if len(self.index.loc[:temp_date]) == 0:
            return None
        members = self.index.loc[:temp_date].iloc[-1].dropna()
        unimembers = list(set(temp_price.columns)&set(members.index))
        temp_price = temp_price[unimembers]
        temp_rf = self.rf[self.rf.index <= temp_date].iloc[-(basewindow + 1):]
        temp_bm = self.bm[self.bm.index <= temp_date].iloc[-(basewindow + 1):]

        minimum_len = len(temp_price)
        temp_factors = {}
        for f in self.factors.keys():
            if Calculator._check_factors(method, f):
                temp_factors[f] = self.factors[f][self.factors[f].index <= temp_date].iloc[-(basewindow + 1):]
                minimum_len = min(minimum_len, len(temp_factors[f]), len(temp_rf), len(temp_bm))

        if minimum_len != len(temp_price):
            return None

        result = self.calc_exp_rtn(method=method
                                , bday=bday
                                , window=window
                                , price=temp_price
                                , rf=temp_rf
                                , bm=temp_bm
                                , factors=temp_factors
                                , view_betas=view_betas
                                )

        if type(result) == pd.Series:
            result.name = temp_price.index[-1]
        if type(result) == pd.DataFrame:
            result.index.name = temp_price.index[-1]

        return result

    def evaluate(self, data:list, window:int=1, resample:str='M', **kwargs) -> pd.DataFrame:
        '''
        예측값과 실제값 간 Loss 계산
        :param data: list of prediction series
        :param window: average window for evaluation
        :param resample: resampling frequency
        :param kwargs: additional arguments
        /corr_method: 'pearson' or 'spearman' for IC and RankIC
        :return: dataframe
        '''
        corr_method = kwargs.get('corr_method', None)

        if resample=='M':
            bday = 12
        elif resample=='W':
            bday = 52

        port_dates = pd.date_range(start=self.price_bd.index[0], end=self.price_bd.index[-1], freq='W-FRI')
        port_dates = pd.Series(port_dates, index=port_dates).resample(resample).last()
        price = self.price_bd.reindex(port_dates.values, method='ffill').dropna(how='all')
        rtn = price.pct_change().dropna(how='all', axis=0)
        rf = self.rf_bd.reindex(port_dates.values, method='ffill').dropna(how='all')

        result = []
        for d in tqdm(data, desc='evaluating'):
            monthly_index = price[d.name:].index
            if len(monthly_index) > 1:
                date = monthly_index[1]
                date_f = monthly_index[window]

                monthly_rf = (1+rf.loc[date:date_f]).mean()**(1/bday)-1
                a = rtn[d.index].loc[date:date_f].mean() - monthly_rf.values

                msfe = self._msfe(a, d)
                ic = self._corr(a, d, method=corr_method)
                rankic = self._rankcorr(a, d, method=corr_method)
                ir = self._ir(a, d)

                result.append([d.name, msfe, ic, rankic, ir])

        result_df = pd.DataFrame(result, columns=['Index', 'MSFE', 'IC', 'Rank_IC', 'IR']).set_index('Index')
        return result_df

    def calc_exp_rtn(self, method: str, bday:int, window:int
                     , price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, factors:Dict
                     , **kwargs): #calc Expected Return
        '''
        method를 기준으로 기대수익률 계산
        :param method: 'mean: 과거수익률 기준'
        :return: dictionary of expected return('prediction') and actual return('actual')
        '''
        view_betas = kwargs.get('view_betas', False)

        if method == 'mean':
            exp_rtn = self._historic_mean(price, rf, window, bday)
        elif method == 'sr2':
            exp_rtn, betas = self._sr2(price, rf, bm, window, bday)
        elif method in MODEL_FACTORS:
            names = MODEL_FACTORS[method]
            missing = [f.upper() for f in names if factors.get(f) is None]
            if missing:
                raise ValueError(f'{", ".join(missing)} should be given for {method} method')
            exp_rtn, betas = self._factor_model(price, rf, bm, factors, names, window, bday)
        else:
            raise ValueError(f'unknown method: {method}')

        exp_rtn = exp_rtn[exp_rtn != 0].dropna()

        # if view_betas is True, return only betas
        if view_betas:
            return betas
        else:
            return exp_rtn

    def _historic_mean(self, price:pd.DataFrame, rf:pd.DataFrame, window:int, bday:int):
        '''
        과거 수익률 평균을 기대수익률로 계산
        :param periods: 과거 수익률 평균을 계산할 기간
        :return: dataframe of expected return
        '''
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        return (rtn - rf.values)[-self.avg_window:].mean(axis=0)

    def _factor_model(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, factors:Dict, names:list, window:int, bday:int):
        '''
        multi-factor 기대수익률: 초과수익률을 bm(-rf) + names 팩터 수익률에 회귀하고, 회귀계수 * 최근 avg_window 평균 팩터수익률을 합산
        (CAPM, FF3, CH4, RV4, FF5, FF6, HS 공통; 모델별 팩터는 MODEL_FACTORS)
        :param factors: factor name -> factor index level (RAW_FACTORS는 수익률로 바꾸지 않고 값 그대로 사용)
        :param names: 회귀에 쓸 팩터 이름 (bm 제외, 순서 = 회귀변수 순서)
        :return: (expected return, betas dataframe[intercept, beta_bm, beta_<name>...])
        '''
        # 1. 초과수익률 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) + 팩터 수익률
        regs = {'bm': bm.pct_change(periods=1, fill_method=None).iloc[-window:] - rf.values}
        for f in names:
            regs[f] = factors[f].iloc[-window:] if f in RAW_FACTORS else factors[f].pct_change(periods=1, fill_method=None).iloc[-window:]

        # 3. multivariate beta 계산
        betas = [Calculator._getBeta(rtn_pure[col], *regs.values()) for col in rtn_pure]
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept'] + [f'beta_{f}' for f in regs])

        # 4. 기대수익률 = intercept + sum(beta * 평균 팩터수익률)
        expected_return = betas_df.intercept
        for f, r in regs.items():
            expected_return = expected_return + betas_df[f'beta_{f}'] * r.iloc[-self.avg_window:].mean().values

        return expected_return, betas_df

    @staticmethod
    def _rev_exp_rtn(price, bm, rf, window, lag, bday):
        # calculate long term reversion
        rtn_rev = price.pct_change(periods=1).iloc[-window -lag: -lag].bfill()
        rtn_bm_lag = bm.pct_change(periods=1).iloc[-window -lag: -lag].bfill()
        Rf = (1 + rf).iloc[-window -lag: -lag] ** (1 / bday) - 1
        rtn_pure_rev = rtn_rev - Rf.values
        rtn_bm_pure_lag = rtn_bm_lag - Rf.values

        rev_betas = []
        for col in rtn_rev.columns:
            beta = Calculator._getBeta(rtn_pure_rev[col], rtn_bm_pure_lag, intercept=True)
            rev_betas.append(beta)
        rev_betas_df = pd.DataFrame(rev_betas, index=rtn_rev.columns, columns=['intercept', 'beta_strev'])

        expected_return_prev = (
                + rev_betas_df.intercept
                + rev_betas_df.beta_strev * rtn_bm_pure_lag.iloc[-1:].mean().values
        )

        return expected_return_prev

    def _sr2(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, window:int, bday:int):
        '''
        short term reversion을 이용한 기대수익률 계산. short term reversion에 해당하는 지표를 추가로 입력받아 계산
        :param price: price of equities
        :param rf: risk free ratio
        :param bm: benchmark price index
        :param window: window size for expected return calculation
        :param bday: bday
        :return: dataframe of expected return
        '''
        #window = 60
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        Rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - Rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - Rf.values

        betas = []
        for col in rtn_pure.columns:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, intercept=True)
            betas.append(beta)

        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm'])

        # calculate long term reversion
        expected_return_prev = Calculator._rev_exp_rtn(price, bm, rf, window, 1, bday)
        de_expected_return_prev = expected_return_prev - expected_return_prev.mean()
        de_rtn_pure = rtn_pure.iloc[-1:].mean().values - rtn_pure.iloc[-1:].mean().values.mean()
        sign1 = de_expected_return_prev / de_rtn_pure

        expected_return_prev = Calculator._rev_exp_rtn(price, bm, rf, window, 2, bday)
        de_expected_return_prev = expected_return_prev - expected_return_prev.mean()
        de_rtn_pure = rtn_pure.iloc[-2:-1].mean().values - rtn_pure.iloc[-2:-1].mean().values.mean()
        sign2 = de_expected_return_prev / de_rtn_pure

        expected_return_prev = Calculator._rev_exp_rtn(price, bm, rf, window, 3, bday)
        de_expected_return_prev = expected_return_prev - expected_return_prev.mean()
        de_rtn_pure = rtn_pure.iloc[-3:-2].mean().values - rtn_pure.iloc[-3:-2].mean().values.mean()
        sign3 = de_expected_return_prev / de_rtn_pure

        sign_rev1 = sign1.apply(lambda x: False if x > 0 else True)
        sign_rev2 = sign2.apply(lambda x: False if x > 0 else True)

        # concatenate signals
        sign_rev = (sign_rev1 | sign_rev2)

        expected_return = (
                + betas_df.intercept
                + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                )

        result = pd.concat([expected_return, sign_rev], axis=1)
        result.columns = ['ER', 'Sign']

        betas_df['beta_sr'] = sign_rev.astype(int)

        #ER_mean = result.ER.mean()

        # stretch minus return
        result['ER_stret'] = result.ER.apply(lambda x: -abs(x)/(1-abs(x)) if x<0 else x)
        # save mean of stretched return
        ER_mean = result.ER_stret.mean()
        # normalize stretched return
        result.ER_stret -= ER_mean

        # apply reversion signal in to return
        result['ER_fit'] = result.apply(lambda x: -x.ER_stret if x.Sign else x.ER_stret, axis=1)
        # unnormalize return
        result.ER_fit += ER_mean
        # contract minus return
        result['ER_fit'] = result.ER_fit.apply(lambda x: -abs(x)*(1-abs(x)) if x<0 else x)

        # 6. 기대수익률 계산
        return result.ER_fit, betas_df

    @staticmethod
    def _msfe(actual, pred):
        '''
        Mean Squared Forecast Error
        :param actual: actual return
        :param pred: predicted return
        :return: MSFE value
        '''
        return ((actual - pred) ** 2).mean()

    @staticmethod
    def _corr(actual, pred, method):
        '''
        Correlation Coefficient
        :param actual: actual return
        :param pred: predicted return
        :param method: 'pearson' or 'spearman'
        :return: Corr value
        '''
        if method=='pearson' or method is None:
            return actual.corr(pred)
        elif method=='spearman':
            return Calculator.spearman_corr(actual, pred)
        else:
            raise ValueError('method should be either pearson or spearman')

    @staticmethod
    def _rankcorr(actual, pred, method):
        '''
        Rank Correlation Coefficient
        :param actual: actual return
        :param pred: predicted return
        :param method: 'pearson' or 'spearman'
        :return: Rank Corr value
        '''
        if method=='pearson':
            return actual.rank().corr(pred.rank())
        elif method=='spearman' or method is None:
            return Calculator.spearman_corr(actual.rank(), pred.rank())
        else:
            raise ValueError('method should be either pearson or spearman')

    @staticmethod
    def _ir(actual, pred):
        '''
        Information Ratio
        :param actual: actual return
        :param pred: predicted return
        :return: IR value
        '''
        return ((actual - pred) / pred.std()).mean()

    @staticmethod
    def _getBeta(y, *x, intercept: bool = True):
        # argument *x to array
        X = np.array(x)
        if len(X.shape) > 2:
            X = X.reshape([X.shape[0], X.shape[1]])
        y = np.array(y)
        y = y.reshape(-1, 1)

        if intercept:
            X = np.vstack([np.ones(len(y)), X]).T
        else:
            X = X.T

        if X.shape[1] == 1:
            return y.sum() / X.sum()
        else:
            return np.ravel(np.linalg.inv(X.T @ X) @ X.T @ y)

    @staticmethod
    def spearman_corr(x, y):

        r = 1 - 6 * np.sum((x - y) ** 2) / (len(x) * (len(x) ** 2 - 1))

        return r
