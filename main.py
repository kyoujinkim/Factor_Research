import pandas as pd
from get_factor_weight import calc_factor_weight, calc_mvo_weight
from backtest_factor import backtest, export_factor
from get_ER import calculate_Expected_Return, calculate_Expected_Return_totalTS
from groupby_ER import groupER
from src.calcEReturn import Calculator

if __name__ == '__main__':
    country_list = ['korea', 'us', 'japan', 'europe']
    # 1. Calculate Factor Index
    ## 1-1. Calculate Factor Index Weight
    '''for country in country_list:
        calc_factor_weight(country, 'm')

    ## 1-2. Calculate Factor Index
    for country in country_list:
        backtest(country
                 , 'm'
                 , factor_list = ['BM', 'SMB', 'HML', 'MOM', 'OP', 'INV'])

    ## export factor index
    for country in country_list:
        export_factor(country
                      , factor_list = ['BM', 'SMB', 'HML', 'MOM', 'OP', 'INV']
                      )

    # 2. Calculate Expected Return
    print('Calculate Expected Return')
    end_date = None # '2024-06-30'
    model_list = [
        ['korea', 'sr2', 3, 60, end_date]
        , ['us', 'ch4', 3, 12, end_date]
        , ['japan', 'sr2', 6, 60, end_date]
        , ['europe', 'capm', 6, 12, end_date]
        ]

    for country_model in model_list:
        snapdate = calculate_Expected_Return(*country_model)

    # 3. Calculate Total Expected Return only for korea
    print('Calculate Total Expected Return')
    model_list = [
        ['korea', 'sr2', 3, 60, end_date]
        , ['us', 'ch4', 3, 12, end_date]
        , ['japan', 'sr2', 6, 60, end_date]
        , ['europe', 'capm', 6, 12, end_date]
        ]
    for country_model in model_list:
        calculate_Expected_Return_totalTS(*country_model)

    print('Calculate MV Optimized Weight for Expected return')
    #snapdate = '2024-11-29'
    # snapdate, country, reverse, rebal_freq, model, avgwindow, datawindow, weightlimit
    model_list = [
        [snapdate, 'korea', 'm', 'sr2', 3, 60, 0.05]
        , [snapdate, 'us', 'm', 'ch4', 3, 12, 0.05]
        , [snapdate, 'japan', 'm', 'sr2', 6, 60, 0.05]
        , [snapdate, 'europe', 'm', 'capm', 6, 12, 0.05]
        ]

    for country_model in model_list:
        calc_mvo_weight(*country_model)'''

    snapdate = '2025-02-28'
    print('Calculate Formatted Expected return')
    model_list = [
        [snapdate, 'korea', 'sr2', 3, 60, 2, 0.9]
        , [snapdate, 'us', 'ch4', 3, 12, 2, 0.7]
        , [snapdate, 'japan', 'sr2', 6, 60, 2, 0.9]
        , [snapdate, 'europe', 'capm', 6, 12, 2, 0.7]
        ]
    for country_model in model_list:
        groupER(*country_model)
