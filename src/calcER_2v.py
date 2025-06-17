import itertools
import os
import warnings
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from pandas._libs.tslibs.offsets import MonthEnd
warnings.simplefilter(action='ignore', category=FutureWarning)

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
            self.rf_bd = rf.fillna(method='ffill') / 100
        if bm is None:
            self.bm_bd = pd.DataFrame(np.ones(len(self.price.index)), index=self.price.index, columns=['bm'])
        else:
            self.bm_bd = bm.fillna(method='ffill')

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

        if resample=='M':
            bday = 12
        elif resample=='W':
            bday = 52
        else:
            bday = 252

        #result = []
        #for i in tqdm(range(basewindow+1, len(self.price))):
        #    result.append(self._iterrun([method, bday, basewindow, window, i, False]))
        iterator = list(itertools.product([method], [bday], [basewindow], [window], range(basewindow+1, len(self.price)+1), [False]))
        result = process_map(self._iterrun, iterator, max_workers=os.cpu_count()-2)

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
        for f in self.factors.keys():
            self.factors[f] = self.factors[f][self.factors[f].index <= temp_date].iloc[-(basewindow + 1):]
            minimum_len = min(minimum_len, len(self.factors[f]))

        if minimum_len != len(temp_price):
            return None

        result = self.calc_exp_rtn(method=method
                                , bday=bday
                                , window=window
                                , price=temp_price
                                , rf=temp_rf
                                , bm=temp_bm
                                , factors=self.factors
                                , view_betas=view_betas
                                )

        result.name = temp_price.index[-1]

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
        elif method == 'capm':
            exp_rtn, betas = self._capm(price, rf, bm, window, bday)
        elif method == 'ff3':
            hml = factors.get('hml', None)
            smb = factors.get('smb', None)
            if hml is None or smb is None:
                raise ValueError('HML and SMB should be given for FF-3F method')
            exp_rtn, betas = self._ff3(price, rf, bm, hml, smb, window, bday)
        elif method == 'ch4':
            hml = factors.get('hml', None)
            smb = factors.get('smb', None)
            mom = factors.get('mom', None)
            if hml is None or smb is None or mom is None:
                raise ValueError('HML, SMB and MOM should be given for CH-4F method')
            exp_rtn, betas = self._ch4(price, rf, bm, hml, smb, mom, window, bday)
        elif method == 'rv4':
            hml = factors.get('hml', None)
            smb = factors.get('smb', None)
            rev = factors.get('rev', None)
            if hml is None or smb is None or rev is None:
                raise ValueError('HML, SMB and REV should be given for RV-4F method')
            exp_rtn, betas = self._rv4(price, rf, bm, hml, smb, rev, window, bday)
        elif method == 'ff5':
            hml = factors.get('hml', None)
            smb = factors.get('smb', None)
            op = factors.get('op', None)
            inv = factors.get('inv', None)
            if hml is None or smb is None or op is None or inv is None:
                raise ValueError('HML, SMB, OP and INV should be given for FF-5F method')
            exp_rtn, betas = self._ff5(price, rf, bm, hml, smb, op, inv, window, bday)
        elif method == 'ff6':
            hml = factors.get('hml', None)
            smb = factors.get('smb', None)
            op = factors.get('op', None)
            inv = factors.get('inv', None)
            mom = factors.get('mom', None)
            if hml is None or smb is None or op is None or inv is None or mom is None:
                raise ValueError('HML, SMB, OP, INV and MOM should be given for FF-6F method')
            exp_rtn, betas = self._ff6(price, rf, bm, hml, smb, op, inv, mom, window, bday)
        elif method == 'sr2':
            exp_rtn, betas = self._sr2(price, rf, bm, window, bday)

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

    def _capm(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, window:int, bday:int):
        '''
        CAPM을 이용한 기대수익률 계산. 과거 12~36개월 평균 시장 수익률 * beta를 통해 주가수익률을 예측할 수 있다는 가정
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return, betas_df

    def _ff3(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, window:int, bday:int):
        '''
        Fama-French 3 Factor Model을 이용한 기대수익률 계산
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_smb_pure = rtn_smb - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return, betas_df

    def _ff5(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, op:pd.DataFrame, inv:pd.DataFrame, window:int, bday:int):
        '''
        Fama-French 5 Factor Model을 이용한 기대수익률 계산
        :param price: price of equities
        :param rf: risk free ratio
        :param bm: benchmark price index
        :param hml: high minus low factor return index
        :param smb: small minus big factor return index
        :param op: operating profitability factor return index
        :param inv: investment factor return index
        :param window: window size for expected return calculation
        :param bday: business day count in a year
        :return: expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_op = op.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_op_pure = rtn_op - rf.values
        rtn_inv = inv.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_inv_pure = rtn_inv - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_op_pure, rtn_inv_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_op', 'beta_inv'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_op * rtn_op_pure.iloc[-self.avg_window:].mean().values
                            + betas_df.beta_inv * rtn_inv_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return, betas_df

    def _ch4(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, mom:pd.DataFrame, window:int, bday:int):
        '''
        Carhart 4 Factor Model을 이용한 기대수익률 계산
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_mom = mom.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_mom_pure = rtn_mom - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_mom_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_mom'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_mom * rtn_mom_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return, betas_df

    def _rv4(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, rev:pd.DataFrame, window:int, bday:int):
        '''
        Short-term reversal 4 Factor Model을 이용한 기대수익률 계산
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_rev = rev.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_rev_pure = rtn_rev - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_rev_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_rev'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_rev * rtn_rev_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return, betas_df

    def _ff6(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, op:pd.DataFrame, inv:pd.DataFrame, mom:pd.DataFrame, window:int, bday:int):
        '''
        Fama-French 6 Factor Model을 이용한 기대수익률 계산

        :param price: price of equities
        :param rf: risk free ratio
        :param bm: bm index
        :param hml: hml index
        :param smb: smb index
        :param op: operating profitability index
        :param inv: investment index
        :param mom: momentum index
        :param window: window size for expected return calculation
        :param bday: bday
        :return: return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-window:]
        rf = (1+rf).iloc[-window:]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb, op, inv, mom 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_op = op.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_op_pure = rtn_op - rf.values
        rtn_inv = inv.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_inv_pure = rtn_inv - rf.values
        rtn_mom = mom.pct_change(periods=1, fill_method=None).iloc[-window:]
        rtn_mom_pure = rtn_mom - rf.values

        # 기존 cov(x,y))/var(x)로는 계산이 어려우므로 행렬곱 방식으로 iteratively 계산
        # 4. multivariate beta 계산
        betas = []
        for col in rtn_pure:
            beta = Calculator._getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_op_pure, rtn_inv_pure, rtn_mom_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_op', 'beta_inv', 'beta_mom'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_op * rtn_op_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_inv * rtn_inv_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_mom * rtn_mom_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
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
        sign_rev3 = sign3.apply(lambda x: False if x > 0 else True)

        # concatenate signals
        sign_rev = (sign_rev1 | sign_rev2 | sign_rev3)

        expected_return = (
                + betas_df.intercept
                + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                )

        result = pd.concat([expected_return, sign_rev], axis=1)
        result.columns = ['ER', 'Sign']

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
