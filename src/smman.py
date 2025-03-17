import multiprocessing.shared_memory as sm
import pyarrow as pa
import pandas as pd

class SharedMemoryManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.shm_name = None
        self.shm = None

    def load_to_shared_memory(self):
        """Load Parquet file into shared memory"""
        table = pd.read_parquet(self.file_path, engine='pyarrow')

        # Serialize to Arrow format for efficient storage
        serialized = pa.ipc.serialize_pandas(table).to_pybytes()

        # Create shared memory
        self.shm = sm.SharedMemory(create=True, size=len(serialized))
        self.shm.buf[:len(serialized)] = serialized
        self.shm_name = self.shm.name

        return self.shm_name

    def read_from_shared_memory(self):
        """Read Parquet data from shared memory"""
        existing_shm = sm.SharedMemory(name=self.shm_name)
        table = pa.ipc.deserialize_pandas(existing_shm.buf[:]).copy()
        existing_shm.close()
        return table

    def cleanup(self):
        """Cleanup shared memory"""
        if self.shm:
            self.shm=None
            self.shm.close()
            self.shm.unlink()