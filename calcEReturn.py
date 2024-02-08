import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
#import plt
import matplotlib.pyplot as plt
from funs import *

class Calculator:
    '''
    계산기 클래스
    '''

    def __init__(self, price:pd.DataFrame, halt: pd.DataFrame, rf: pd.DataFrame=None, bm: pd.DataFrame=None):
        self.price_bd = price
        self.halt_bd = halt
        if rf is None:
            #set rf as 0 if not given
            self.rf_bd = pd.DataFrame(0, index=halt.index)
        else:
            self.rf_bd = rf.fillna(method='ffill') / 100 #since rf is given as percentage
        if bm is None:
            self.bm_bd = pd.DataFrame(1, index=halt.index)
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
        halt = (self.halt_bd[(self.halt_bd.index <= temp_price.index[-1]) & (
                    self.halt_bd.index >= temp_price.index[-window])]
                .dropna(axis=1).sum())
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

        return [temp_price.index[-1]
            , self.calc_exp_rtn(method=method
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
                ]

    def evaluate(self, data:list):

        result = []
        for d in tqdm(data, desc='evaluating'):
            msfe = self._eval('msfe', d[1]['actual'], d[1]['prediction'])
            ic = self._eval('ic', d[1]['actual'], d[1]['prediction'])
            rankic = self._eval('rankic', d[1]['actual'], d[1]['prediction'])
            ir = self._eval('ir', d[1]['actual'], d[1]['prediction'])

            result.append([d[0], msfe, ic, rankic, ir])

        return result

    def _eval(self, method: str, actual, pred):
        '''
        예측값과 실제값 간 Loss 계산
        :param method: 'msfe: Mean Squared Forecast Error'
        :param actual: actual return
        :param pred: predicted return
        :return: loss value
        '''
        if method == 'msfe':
            return self._msfe(actual, pred)
        elif method == 'ic':
            return self._ic(actual, pred)
        elif method == 'ir':
            return self._ir(actual, pred)

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
        actual = (price.pct_change(periods=1, fill_method=None).iloc[-1][exp_rtn.index]
                  - ((1+rf.iloc[-1])**(1/bday)-1).values)
        return {'prediction': exp_rtn, 'actual': actual}

    def _historic_mean(self, price:pd.DataFrame, rf:pd.DataFrame, window:int, bday:int):
        '''
        과거 수익률 평균을 기대수익률로 계산
        :param periods: 과거 수익률 평균을 계산할 기간
        :return: dataframe of expected return
        '''
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rf = (1+rf).iloc[-(window + 1):-1]**(1/bday)-1
        return (rtn - rf.values).mean(axis=0)

    def _capm(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, window:int, bday:int):
        '''
        CAPM을 이용한 기대수익률 계산. 과거 12~36개월 평균 시장 수익률 * beta를 통해 주가수익률을 예측할 수 있다는 가정
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rf = (1+rf).iloc[-(window + 1):-1]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. beta 계산
        cov = pd.concat([rtn_pure, rtn_bm_pure], axis=1).cov()
        diag = np.diag(cov)
        beta = cov.iloc[-1, :-1] / diag[1:]

        # 4. alpha 계산
        alpha = rtn_pure.mean(axis=0) - beta * rtn_bm_pure.mean(axis=0).values

        # 4. 기대수익률 계산
        return alpha + beta * rtn_bm_pure.iloc[-self.avg_window:].mean().values

    def _ff3(self, price:pd.DataFrame, rf:pd.DataFrame, bm:pd.DataFrame, hml:pd.DataFrame, smb:pd.DataFrame, window:int, bday:int):
        '''
        Fama-French 3 Factor Model을 이용한 기대수익률 계산
        :param periods: 기대수익률을 계산할 기간
        :return: dataframe of expected return
        '''
        # 1. 과거 12~36개월 평균 시장 수익률(-rf) 계산
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rf = (1+rf).iloc[-(window + 1):-1]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
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
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rf = (1+rf).iloc[-(window + 1):-1]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_op = op.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_op_pure = rtn_op - rf.values
        rtn_inv = inv.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
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
        rtn = price.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rf = (1+rf).iloc[-(window + 1):-1]**(1/bday)-1
        rtn_pure = rtn - rf.values

        # 2. bm 수익률(-rf) 계산
        rtn_bm = bm.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_bm_pure = rtn_bm - rf.values

        # 3. hml, smb 수익률(-rf) 계산
        rtn_hml = hml.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_hml_pure = rtn_hml - rf.values
        rtn_smb = smb.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
        rtn_smb_pure = rtn_smb - rf.values
        rtn_mom = mom.pct_change(periods=1, fill_method=None).iloc[-(window + 1):-1]
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

    def _ic(self, actual, pred):
        '''
        Information Coefficient
        :param actual: actual return
        :param pred: predicted return
        :return: IC value
        '''
        return actual.corr(pred)

    def _rankic(self, actual, pred):
        '''
        Rank Information Coefficient
        :param actual: actual return
        :param pred: predicted return
        :return: Rank IC value
        '''
        return actual.rank().corr(pred.rank())

    def _ir(self, actual, pred):
        '''
        Information Ratio
        :param actual: actual return
        :param pred: predicted return
        :return: IR value
        '''
        return ((actual - pred) / pred.std()).mean()


def get_eval_df(filepath):
    c = load_pickle(filepath)

    p = cal.evaluate(c)
    p_df = pd.DataFrame(p).set_index(0)

    return p_df


if __name__ == '__main__':

    country = 'europe' #'us' or 'korea' or 'japan' or 'europe

    rf_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_riskfree.csv', dtype=float, index_col=0, parse_dates=True)
    price_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    halt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_halt.csv', dtype=float, index_col=0, parse_dates=True)
    mkt_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_marketvalue.csv', dtype=float, index_col=0, parse_dates=True)
    bv_df = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_bookvalue.csv', dtype=float, index_col=0, parse_dates=True)
    bm_df = pd.read_csv(f'./output/{country}/bm/price_bm_m.csv', dtype=float, index_col=0, parse_dates=True)
    smb_df = pd.read_csv(f'./output/{country}/bm/price_smb_m.csv', dtype=float, index_col=0, parse_dates=True)
    hml_df = pd.read_csv(f'./output/{country}/bm/price_hml_m.csv', dtype=float, index_col=0, parse_dates=True)
    mom_df = pd.read_csv(f'./output/{country}/bm/price_mom_m.csv', dtype=float, index_col=0, parse_dates=True)
    op_df = pd.read_csv(f'./output/{country}/bm/price_op_m.csv', dtype=float, index_col=0, parse_dates=True)
    inv_df = pd.read_csv(f'./output/{country}/bm/price_inv_m.csv', dtype=float, index_col=0, parse_dates=True)

    # Unify index of all data to max lower bound
    basedate = findBaseDate(rf_df, price_df, halt_df, mkt_df, bm_df, smb_df, hml_df, mom_df, op_df, inv_df)
    rf_df = rf_df[rf_df.index >= basedate]
    price_df = price_df[price_df.index >= basedate]
    halt_df = halt_df[halt_df.index >= basedate]
    mkt_df = mkt_df[mkt_df.index >= basedate]
    bm_df = bm_df[bm_df.index >= basedate]
    # also unify index for factors
    smb_df = smb_df[smb_df.index >= basedate]
    hml_df = hml_df[hml_df.index >= basedate]
    mom_df = mom_df[mom_df.index >= basedate]
    op_df = op_df[op_df.index >= basedate]
    inv_df = inv_df[inv_df.index >= basedate]

    cal = Calculator(price=price_df, halt=halt_df, rf=rf_df, bm=bm_df)
    '''
    datawindow = 60 # 사용할 데이터의 window
    basewindow = 61 # 넉넉하게 부여
    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, avg_window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_3m_60m.pkl')
    c = cal.run(method='ff3', resample='M', basewindow=basewindow, window=datawindow, hml=hml_df, smb=smb_df, avg_window=12)
    save_pickle(c, f'./output/{country}/Expected_Return/ff3_12m_60m.pkl')


    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=3)
    save_pickle(c, f'./output/{country}/Expected_Return/mean_3m_60m.pkl')
    c = cal.run(method='mean', resample='M', basewindow=basewindow, window=12)
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
    save_pickle(c, f'./output/{country}/Expected_Return/ff5_12m_60m.pkl')
    quit()'''

    mean_3m_60m = get_eval_df(f'./output/{country}/Expected_Return/mean_3m_60m.pkl')
    mean_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/mean_12m_60m.pkl')
    capm_3m_60m = get_eval_df(f'./output/{country}/Expected_Return/capm_3m_60m.pkl')
    capm_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/capm_12m_60m.pkl')
    ff3_3m_60m = get_eval_df(f'./output/{country}/Expected_Return/ff3_3m_60m.pkl')
    ff3_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/ff3_12m_60m.pkl')
    ch4_3m_60m = get_eval_df(f'./output/{country}/Expected_Return/ch4_3m_60m.pkl')
    ch4_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/ch4_12m_60m.pkl')
    ff5_3m_60m = get_eval_df(f'./output/{country}/Expected_Return/ff5_3m_60m.pkl')
    ff5_12m_60m = get_eval_df(f'./output/{country}/Expected_Return/ff5_12m_60m.pkl')

    print('mean_3m_60m : \n', mean_3m_60m.mean())
    print('mean_12m_60m : \n', mean_12m_60m.mean())
    print('capm_3m_60m : \n', capm_3m_60m.mean())
    print('capm_12m_60m : \n', capm_12m_60m.mean())
    print('ff3_3m_60m : \n', ff3_3m_60m.mean())
    print('ff3_12m_60m : \n', ff3_12m_60m.mean())
    print('ch4_3m_60m : \n', ch4_3m_60m.mean())
    print('ch4_12m_60m : \n', ch4_12m_60m.mean())
    print('ff5_3m_60m : \n', ff5_3m_60m.mean())
    print('ff5_12m_60m : \n', ff5_12m_60m.mean())

    print('done')

    '''
    Note
    - 보다 장기 window의 regression 베타와 장기 mean vector를 사용하면 할수록 CAPM이 단순 과거 수익률 평균보다 낮은 MSFE와 높은 IR을 기록
    - FF3, CH4 모델은 CAPM보다 높은 MSFE와 낮은 IR을 기록
    - FF3, CH4 모델은 장기 window의 regression 베타와 장기 mean vector를 사용하면 할수록 MSFE가 높아지고 IR이 낮아짐
    - 다만, IC와의 관계는 불확실, 한국은 (-)의 관계가 나오지만, 미국은 (+)의 관계가 나옴
    - 한국은 FF3 < CH4 관계가 성립되나, 미국은 다소 불확실함(MSFE 기준)
    #하 시발 어렵다
        
    '''