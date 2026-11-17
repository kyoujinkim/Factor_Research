'''
Single place for paths and per-market settings (main_v2.py, updateData.py and src/loadData.py read from here).
'''

DB_PATH = r'C:\Users\NHWM\Factor_DB'
CONFIG_INI = 'D:/config.ini'    # Refinitiv / Datastream credentials (updateData.py)

# factor name -> score built by DataProcessor.build_data (see src/factorLib.py for sign conventions)
# the order is also the column order of output/0.export/{country}/factor_hs.csv
FACTOR_SCORES = {
    'BM': 'amktcap',
    'HML': 'value',
    'SMB': 'size',
    'MOM': 'momentum',
    'OP': 'profitability',
    'INV': 'investment',
    'REV': 'reversal',
}

# factor indices fed into the ER regressions (Calculator picks the ones its model uses)
ER_FACTORS = ['bm', 'hml', 'smb', 'mom', 'op', 'inv']
# factors whose current weights are exported as factor_ER by groupER
EXPORT_FACTORS = ['BM', 'HML', 'INV', 'MOM', 'OP', 'SMB']

# model/avg_window/window: calculate_Expected_Return; mktpct: groupER top/bottom filter (-1 = no filter)
MARKETS = {
    'us':     dict(index='RAY',    currency='USD', currency_ds='U$', model='ch4',  avg_window=3,  window=12, mktpct=0.7),
    'europe': dict(index='SXXP',   currency='EUR', currency_ds='E',  model='capm', avg_window=6,  window=12, mktpct=-1),
    'japan':  dict(index='TPX500', currency='JPY', currency_ds='Y',  model='capm', avg_window=3,  window=60, mktpct=-1),
    'korea':  dict(index='KRX500', currency='KRW', currency_ds='KW', model='capm', avg_window=3,  window=60, mktpct=-1),
    'china':  dict(index='MXCN',   currency='CNY', currency_ds='CH', model='ff3',  avg_window=12, window=60, mktpct=-1),
}
