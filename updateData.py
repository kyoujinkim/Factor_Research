from configparser import ConfigParser

from src.data_update_bulk import DataDownloader
from src.ravel_data import DataProcessor

# Load App Key from config file
config = ConfigParser()
config.read('D:/config.ini')
app_key = config['main']['api_key']
id = config['dsws']['id']
pw = config['dsws']['pw']

taglist = ['name', 'sector', 'issuename', 'country', 'currency', 'pcurrency']
datalist = ['closeprice', 'returns', 'fyshares', 'mktcap', 'dividend', 'earnings', 'bookvalue', 'totalassets', 'freefloatpct']

prev_months = 12 * 5 + 3
post_months = 3

dD = DataDownloader(app_key, id, pw, 'D:/Factor_DB', 'equity_master.db', 'data_master.json')
dP = DataProcessor('D:/Factor_DB', 'equity_master.db', 'data_master.json')
job = 'Done'

for index, curr, curr_ds in [
              ['RAY','USD', 'U$'],
              ['SXXP', 'EUR', 'E'],
              ['TPX500', 'JPY', 'Y'],
              ['KRX500', 'KRW', 'KW'],
              ['MXCN', 'CNY', 'CH']
              ]:

    dD.assign_index(index, exist=False)
    print(dD.index)

    for tag in taglist:
        job = dD.process_tag(tag, force_update=True)
        print(f'{tag} done {job}')

    for data in datalist:
        job = dD.process(prev_months=prev_months, post_months=post_months, data=data, currency=curr, currency_ds=curr_ds, force_update=False)
        print(f'{index} {data} done {job}')

    for data in datalist:
        job = dP.process(data, force_update=False)
        print(f'{data} done {job}')

dD.equity_master.close()
dP.equity_master.close()