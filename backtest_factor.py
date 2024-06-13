import warnings
import pandas as pd
import bt


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def run_bt(weight_path, price_bd, div_bd):
    weight = pd.read_csv(weight_path, index_col=0, parse_dates=True)
    strat_name = weight_path.split('/')[-1].split('.')[0]

    price_bt = price_bd[price_bd.index >= weight.index[0]][weight.columns]
    div_bt = div_bd[div_bd.index >= weight.index[0]][weight.columns]

    price_bt = price_bt.ffill()

    securities = [bt.CouponPayingSecurity(name, fixed_income=False) for name in weight.columns.values]
    s = bt.Strategy(strat_name
                    , [bt.algos.WeighTarget(weight)
                        , bt.algos.Rebalance()
                       ]
                    , children=securities
                    )

    t = bt.Backtest(s, price_bt
                    , integer_positions=False
                    #, initial_capital=100000000000# 매우 큰값으로 설정. 결과값에는 영향이 없으나, 단주/ 커미션 부족 등의 에러 방지
                    #, commissions=lambda q, p: abs(q) * p * 0.00215
                    , additional_data={'coupons': div_bt}
                    , progress_bar=True
                    ,
                    )

    return t

def load_weight_and_backtest(price_bd, div_bd, weight_path, output_path):
    res = bt.run(run_bt(weight_path, price_bd, div_bd))
    res.prices.to_csv(output_path)

def export_factor(country='us', rebal_freq='m', factor_list=['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV']):

    result = []
    for factor in factor_list:
        factor_data = pd.read_csv(f'./output/{country}/bm/{factor}_{rebal_freq}.csv', index_col=0, parse_dates=True)
        result.append(factor_data)

    result = pd.concat(result, axis=1)

    result.to_csv(f'output/0.export/{country}/factor_ts.csv')

    return True

def backtest(country='us', rebal_freq='m', factor_list=['BM', 'HML', 'SMB', 'MOM', 'OP', 'INV']):

    '''load data'''
    price_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_price.csv', dtype=float, index_col=0, parse_dates=True)
    div_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_dividend.csv', dtype=float, index_col=0, parse_dates=True)

    # consider Dividend Date, adjust dividend date
    div_bd = div_bd.shift(-1).fillna(0)

    # cai 불러오기
    cai_bd = pd.read_csv(f'D:/Factor_DB/{country}/{country}_equity_cai.csv', index_col=0, parse_dates=True)
    cai_bd = cai_bd.reindex(price_bd.index, method='bfill').ffill()
    cai_bd = cai_bd.copy() / cai_bd.iloc[-1]
    cai_bd = cai_bd.fillna(1)

    # cai를 기준으로 share관련 지표들 수정
    price_bd *= cai_bd
    div_bd *= cai_bd

    for factor in factor_list:
        weight_path = f'./weight/{country}/factor_weight/{factor}.csv'
        output_path = f'./output/{country}/bm/{factor}_{rebal_freq}.csv'
        load_weight_and_backtest(price_bd, div_bd, weight_path, output_path)

    print('done')