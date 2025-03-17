import shutil
from glob import glob

import pandas as pd
from src.funs import *
import src.factorLib as flib
from src.calcWeight_2v import WeightBuilder
import src.loadData as ld


def reindex_all(main, *args):
    basedate = findBaseDate(main, *args)
    main = main[main.index >= basedate]
    for arg in args:
        arg = arg[arg.index >= basedate][main.columns]
    return main, args


if __name__ == '__main__':

    dp = ld.DataProcessor()
    dp.turnon_cache()

    # 1. index 선택
    index = 'RAY'
    weighting_method = 'equal'

    # 2. index별로 저장되어있는 index_master 불러오기
    index_master = pd.read_parquet(f'D:/Factor_DB/members/master_{index}.parquet', engine='pyarrow')
    #curr_index = index_master.iloc[-1:].dropna(axis=1)
    dp.load_stocks(index_master.columns)

    bpath = './cache/us'
    #shutil.rmtree(bpath, ignore_errors=True)
    #os.makedirs(bpath, exist_ok=True)


    factor_list = ['momentum', 'value', 'size', 'investment', 'profitability']
    data_list = ['mktcap']
    for factor in factor_list:
        # if factor is designated, build data for factor
        data_list.append(factor)

    if weighting_method == 'optmvo':
        data_list.append('return')
    else:
        pass

    data_list = list(set(data_list))

    # build data
    #for data in tqdm(data_list):
    #    dp.build_data(bpath, data, None, None)

    dp.turnoff_cache()

    wb = WeightBuilder(index_master, rebal_freq='m', data_freq='W-FRI')

    for factor in factor_list:
        w = wb.run(start=None
                   , end=None
                   , weighting_method=weighting_method
                   , factor=factor
                   , base_path=bpath
                   , workers=os.cpu_count() - 3
                   , active_multiproc=True)

        w.to_csv(f'tmp{factor}.csv')
