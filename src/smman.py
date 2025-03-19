import multiprocessing.shared_memory as sm
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pandas as pd

def call_shared_memory(path):
    '''
    df = pd.read_parquet('./cache/us/mktcap.parquet')
    np_array = df.to_numpy()
    shm = smm.SharedMemory(create=True, size=np_array.nbytes)
    shm_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    np.copyto(shm_array, np_array)

    new_shm = smm.SharedMemory(name=shm.name)
    array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=new_shm.buf)

    즉 shm의 이름을 통해 shm을 불러오고
    shm에 저장된 shape와 dtype과 shm의 buffer를 통해 데이터를 호출해야한다.
    이 때, 인덱스나 컬럼은 저장되지않으므로 이도 따로 메타데이터로 저장해야한다.
    따라서 다음과 같은 클래스를 구상할 수 있다.
    우선 parquet의 path를 지정하면, 해당 parquet을 shared memory에 저장하고
    해당 shared memory의 이름과 shape, dtype, index, columns를 저장하는 클래스를 구상한다.
    그리고 해당 정보를 반환하는 함수를 지닌다.
    '''
    df = pd.read_parquet(path)
    np_array = df.to_numpy()
    shm = sm.SharedMemory(create=True, size=np_array.nbytes)
    shm_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    np.copyto(shm_array, np_array)

    # return dict
    return {'shm':shm, 'shape': np_array.shape, 'dtype': np_array.dtype, 'index': df.index, 'columns': df.columns}

def load_shared_memory(info):
    new_shm = sm.SharedMemory(name=info['shm'].name)
    array = np.ndarray(info['shape'], dtype=info['dtype'], buffer=new_shm.buf)
    df = pd.DataFrame(array, index=info['index'], columns=info['columns'])
    new_shm.close()

    return df

from multiprocessing.managers import SharedMemoryManager

class test:
    memory = {'a':1, 'b':2}

smm = SharedMemoryManager()
smm.start()
