from configparser import ConfigParser

import config
from src.data_update_bulk import DataDownloader
from src.ravel_data import RawDataWriter

# Load App Key from config file
ini = ConfigParser()
ini.read(config.CONFIG_INI)
app_key = ini['main']['api_key']
id = ini['dsws']['id']
pw = ini['dsws']['pw']

taglist = ['name', 'sector', 'issuename', 'country', 'currency', 'pcurrency']
datalist = ['closeprice', 'returns', 'fyshares', 'mktcap', 'dividend', 'earnings', 'bookvalue', 'totalassets', 'freefloatpct']

prev_months = 12 * 5 + 3
post_months = 3

dD = DataDownloader(app_key, id, pw, config.DB_PATH, 'equity_master.db', 'data_master.json')
dP = RawDataWriter(config.DB_PATH, 'equity_master.db', 'data_master.json')
job = 'Done'

for m in config.MARKETS.values():
    index, curr, curr_ds = m['index'], m['currency'], m['currency_ds']

    dD.assign_index(index, exist=False)
    print(dD.index)

    for tag in taglist:
        job = dD.process_tag(tag, force_update=False)
        print(f'{tag} done {job}')

    for data in datalist:
        job = dD.process(prev_months=prev_months, post_months=post_months, data=data, currency=curr, currency_ds=curr_ds, force_update=False)
        print(f'{index} {data} done {job}')

    for data in datalist:
        job = dP.process(data, force_update=False)
        print(f'{data} done {job}')

dD.equity_master.close()
dP.equity_master.close()
