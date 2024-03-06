import itertools

import numpy as np
import pandas as pd
from pandas._libs.tslibs.offsets import MonthEnd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
#import plt
import matplotlib.pyplot as plt
from src.funs import *

class Calculator:
    '''
    계산기 클래스
    '''

    def __init__(self, price:pd.DataFrame, halt: pd.DataFrame=None, rf: pd.DataFrame=None, bm: pd.DataFrame=None):
        self.price_bd = price
        if halt is None:
            self.halt_bd = pd.DataFrame(0, index=price.index, columns=price.columns)
        else:
            self.halt_bd = halt
        if rf is None:
            #set rf as 0 if not given
            self.rf_bd = pd.DataFrame(0, index=halt.index)
        else:
            # since rf is given as percentage
            self.rf_bd = rf.fillna(method='ffill') / 100
        if bm is None:
            self.bm_bd = pd.DataFrame(np.ones(len(self.halt_bd.index)), index=self.halt_bd.index, columns=['bm'])
        else:
            self.bm_bd = bm.fillna(method='ffill')

    def run(self, method:str, resample:str='M', basewindow:int=24, window:int=12, **kwargs) -> list:
        '''
        날짜, 예측값, 실제값의 리스트를 반환하는 함수
        :param method: 예측값을 계산하는 방법
        :param resample: 수익률 계산시에 사용할 리샘플링 날짜 단위
        :param basewindow: 수익률 계산에 사용할 기간(넉넉하게)
        :param window: 예측값 계산에 사용할 날짜 기간
        :param kwargs: 추가적인 인자
            SML: small minus large factor return index
            HML: High minus Low factor return index
            avg_window: historical mean return when calculate exp_return
        :return: 날짜, 예측값, 실제값의 리스트
        '''

        self.price = self.price_bd.resample(resample).last()
        self.rf = self.rf_bd.resample(resample).last()
        self.bm = self.bm_bd.resample(resample).last()

        #assign kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)
            #if value is dataframe, resample it
            if isinstance(value, pd.DataFrame):
                setattr(self, key, value.resample(resample).last())
        self.avg_window = kwargs.get('avg_window', 12)

        if resample=='M':
            bday = 12
        elif resample=='W':
            bday = 52

        #result = []
        #for i in tqdm(range(basewindow+1, len(Calculator.price))):
        #    result.append(self.__iterrun([method, bday, basewindow, window, i]))
        iterator = list(itertools.product([method], [bday], [basewindow], [window], range(basewindow+1, len(self.price))))
        result = process_map(self._iterrun, iterator, max_workers=os.cpu_count()-2)

        return result

    def _iterrun(self, var):

        method, bday, basewindow, window, i = var

        temp_price = self.price.iloc[i - (basewindow + 1):i]
        halt = self.halt_bd[self.halt_bd.index <= temp_price.index[-1]][-window:].dropna(axis=1).sum()
        halt = halt[halt == 0]
        temp_price = temp_price[halt.index]
        temp_rf = self.rf.iloc[i - (basewindow + 1):i]
        temp_bm = self.bm.iloc[i - (basewindow + 1):i]

        try: temp_hml = self.hml.iloc[i - (basewindow + 1):i]
        except: temp_hml = None
        try: temp_smb = self.smb.iloc[i - (basewindow + 1):i]
        except: temp_smb = None
        try: temp_mom = self.mom.iloc[i - (basewindow + 1):i]
        except: temp_mom = None
        try: temp_op = self.op.iloc[i - (basewindow + 1):i]
        except: temp_op = None
        try: temp_inv = self.inv.iloc[i - (basewindow + 1):i]
        except: temp_inv = None

        result = self.calc_exp_rtn(method=method
                                , bday=bday
                                , window=window
                                , price=temp_price
                                , rf=temp_rf
                                , bm=temp_bm
                                , hml=temp_hml
                                , smb=temp_smb
                                , mom=temp_mom
                                , op=temp_op
                                , inv=temp_inv
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

        rtn = self.price_bd.resample('M').last().pct_change().dropna(how='all', axis=0)
        rf = self.rf_bd.resample('M').last()

        result = []
        for d in tqdm(data, desc='evaluating'):
            d = -d
            date = d.name + MonthEnd(1)
            date_f = d.name + MonthEnd(window)

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
                     , price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame
                     , **kwargs): #calc Expected Return
        '''
        method를 기준으로 기대수익률 계산
        :param method: 'mean: 과거수익률 기준'
        :return: dictionary of expected return('prediction') and actual return('actual')
        '''
        if method == 'mean':
            exp_rtn = self._historic_mean(price, rf, window, bday)
        elif method == 'capm':
            exp_rtn = self._capm(price, rf, bm, window, bday)
        elif method == 'ff3':
            hml = kwargs.get('hml', None)
            smb = kwargs.get('smb', None)
            if hml is None or smb is None:
                raise ValueError('HML and SMB should be given for FF-3F method')
            exp_rtn = self._ff3(price, rf, bm, hml, smb, window, bday)
        elif method == 'ch4':
            hml = kwargs.get('hml', None)
            smb = kwargs.get('smb', None)
            mom = kwargs.get('mom', None)
            if hml is None or smb is None or mom is None:
                raise ValueError('HML, SMB and MOM should be given for CH-4F method')
            exp_rtn = self._ch4(price, rf, bm, hml, smb, mom, window, bday)
        elif method == 'ff5':
            hml = kwargs.get('hml', None)
            smb = kwargs.get('smb', None)
            op = kwargs.get('op', None)
            inv = kwargs.get('inv', None)
            if hml is None or smb is None or op is None or inv is None:
                raise ValueError('HML, SMB, OP and INV should be given for FF-5F method')
            exp_rtn = self._ff5(price, rf, bm, hml, smb, op, inv, window, bday)

        exp_rtn = exp_rtn[exp_rtn != 0].dropna()

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
            beta = getBeta(rtn_pure[col], rtn_bm_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return

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
            beta = getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return

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
            beta = getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_op_pure, rtn_inv_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_op', 'beta_inv'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_op * rtn_op_pure.iloc[-self.avg_window:].mean().values
                            + betas_df.beta_inv * rtn_inv_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return

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
            beta = getBeta(rtn_pure[col], rtn_bm_pure, rtn_hml_pure, rtn_smb_pure, rtn_mom_pure)
            betas.append(beta)
        betas_df = pd.DataFrame(betas, index=rtn_pure.columns, columns=['intercept', 'beta_bm', 'beta_hml', 'beta_smb', 'beta_mom'])
        expected_return = (betas_df.intercept
                           + betas_df.beta_bm * rtn_bm_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_hml * rtn_hml_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_smb * rtn_smb_pure.iloc[-self.avg_window:].mean().values
                           + betas_df.beta_mom * rtn_mom_pure.iloc[-self.avg_window:].mean().values)

        # 6. 기대수익률 계산
        return expected_return

    def _msfe(self, actual, pred):
        '''
        Mean Squared Forecast Error
        :param actual: actual return
        :param pred: predicted return
        :return: MSFE value
        '''
        return ((actual - pred) ** 2).mean()

    def _corr(self, actual, pred, method):
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
            return spearman_corr(actual, pred)
        else:
            raise ValueError('method should be either pearson or spearman')

    def _rankcorr(self, actual, pred, method):
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
            return spearman_corr(actual.rank(), pred.rank())
        else:
            raise ValueError('method should be either pearson or spearman')

    def _ir(self, actual, pred):
        '''
        Information Ratio
        :param actual: actual return
        :param pred: predicted return
        :return: IR value
        '''
        return ((actual - pred) / (actual-pred).std()).mean()