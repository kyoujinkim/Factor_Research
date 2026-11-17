'''
RawStore: a disposable, consolidated copy of the per-stock parquet files (db/{instrument}_{key}.parquet).

The per-stock files stay the source of truth (one stock can be re-downloaded on its own).
The store only removes the cost of opening ~10,000 small files per key, and can be deleted and rebuilt at any time.

    {root}/{key}.parquet            long table: date, instrument, {key} [, fydate], sorted by instrument
    {root}/{key}.manifest.json      {instrument: [mtime_ns, size]} of the per-stock file each row group was built from

Two writers keep it in sync:
    - ensure(): on read, compare the manifest with the files on disk and re-read only new/changed stock files
    - upsert(): on ingestion, replace the rows of the stocks that were just written (rows are already in memory)
Anything that goes wrong falls back to scanning the per-stock files, so the result never depends on the store.

StockFiles is the one place that lists the per-stock files; both ensure() and the build_data cache signature
(loadData.DataProcessor) read the same listing, so they always judge "has the source changed" from the same state.
'''
import hashlib
import json
import os
import time
from contextlib import contextmanager

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


class StockFiles:
    '''
    State of the per-stock files: one directory listing gives (mtime_ns, size) of every file of every key.
    The listing is taken once per instance and reused; refresh() makes the next call list the directory again.
    '''
    def __init__(self, dbdir: str):
        self.dbdir = dbdir.replace('\\', '/')
        self._stats = None

    def scan(self) -> dict:
        '''{key: {instrument: [mtime_ns, size]}}'''
        if self._stats is None:
            stats = {}
            with os.scandir(self.dbdir) as entries:
                for e in entries:
                    if not e.name.lower().endswith('.parquet'):
                        continue
                    instrument, _, key = e.name[:-len('.parquet')].partition('_')
                    if key:
                        st = e.stat()
                        stats.setdefault(key.lower(), {})[instrument] = [st.st_mtime_ns, st.st_size]
            self._stats = stats
        return self._stats

    def refresh(self):
        self._stats = None

    def signature(self, instruments, keys) -> str:
        '''hash of the files of these instruments for these keys: changes when any of them is added, rewritten or removed'''
        stats = self.scan()
        entries = sorted(f'{i}|{k}|{m}|{s}' for k in keys for i, (m, s) in stats.get(k, {}).items() if i in instruments)
        return hashlib.sha1('\n'.join(entries).encode()).hexdigest()


class RawStore:
    LOCK_STALE_SEC = 1800
    ROW_GROUP_SIZE = 1_000_000

    def __init__(self, root: str, dbdir: str, files: StockFiles = None):
        self.root = root.replace('\\', '/')
        self.dbdir = dbdir.replace('\\', '/')
        self.files = files or StockFiles(self.dbdir)

    def _path(self, key):
        return f'{self.root}/{key}.parquet'

    def _manifest_path(self, key):
        return f'{self.root}/{key}.manifest.json'

    def _lock_path(self, key):
        return f'{self.root}/{key}.lock'

    def _scan(self, key) -> dict:
        '''{instrument: [mtime_ns, size]} of every per-stock file of this key (from the shared listing)'''
        return self.files.scan().get(key, {})

    def _load_manifest(self, key) -> dict:
        try:
            with open(self._manifest_path(key), 'r') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _write_manifest(self, key, manifest):
        path = self._manifest_path(key)
        with open(f'{path}.tmp', 'w') as f:
            json.dump(manifest, f)
        os.replace(f'{path}.tmp', path)

    @staticmethod
    def _readable(path) -> bool:
        try:
            pq.read_metadata(path)
            return True
        except Exception:
            return False

    @contextmanager
    def _lock(self, key):
        '''yields True if this process may write the store of `key`, False if another writer holds it'''
        os.makedirs(self.root, exist_ok=True)
        path = self._lock_path(key)
        fd = None
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(path) > self.LOCK_STALE_SEC:
                        os.remove(path)     # left behind by a crashed writer
                        continue
                except OSError:
                    pass
                break
        if fd is None:
            yield False
            return
        try:
            yield True
        finally:
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass

    @staticmethod
    def _empty(key):
        return pa.table({'date': pa.array([], pa.timestamp('ns')),
                         'instrument': pa.array([], pa.string()),
                         key: pa.array([], pa.float64())})

    @staticmethod
    def _normalize(tbl, key):
        '''fixed column set/types, so rows from files and rows from memory concatenate cleanly; rows without a date never match a date filter'''
        if not {'date', 'instrument', key}.issubset(tbl.column_names):
            return RawStore._empty(key)
        tbl = tbl.select([c for c in ('date', 'instrument', key, 'fydate') if c in tbl.column_names])
        for name, typ in (('date', pa.timestamp('ns')), ('instrument', pa.string()), (key, pa.float64())):
            if tbl.schema.field(name).type != typ:
                tbl = tbl.set_column(tbl.schema.get_field_index(name), name, tbl[name].cast(typ))
        return tbl.filter(pc.is_valid(tbl['date']))

    def _read_files(self, key, instruments):
        files = [f'{self.dbdir}/{i}_{key}.parquet' for i in sorted(instruments)]
        res = duckdb.sql(f"SELECT * FROM read_parquet({files}, union_by_name=true)").arrow()
        if hasattr(res, 'read_all'):
            res = res.read_all()
        return self._normalize(res, key)

    def _apply(self, key, manifest, new_tbl, new_stats, drop_ids):
        '''replace the rows of new_stats' instruments (and remove drop_ids) in the store, then publish the manifest.
        `manifest` must already be validated against the store file (empty -> the store is rebuilt from new_tbl alone).'''
        os.makedirs(self.root, exist_ok=True)
        path = self._path(key)
        replace_ids = set(new_stats) | set(drop_ids)

        kept = None
        if manifest:
            kept = pq.read_table(path)
            if replace_ids:
                mask = pc.is_in(kept['instrument'], value_set=pa.array(sorted(replace_ids), pa.string()))
                kept = kept.filter(pc.invert(mask))

        if kept is not None and kept.num_rows and new_tbl.num_rows:
            # stable sort restores the per-stock file order (instrument, then original row order)
            merged = pa.concat_tables([kept, new_tbl], promote_options='default').sort_by('instrument')
        elif kept is not None and kept.num_rows:
            merged = kept
        else:
            merged = new_tbl

        tmp = f'{path}.tmp'
        try:
            pq.write_table(merged, tmp, row_group_size=self.ROW_GROUP_SIZE)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        new_manifest = {k: v for k, v in manifest.items() if k not in drop_ids}
        new_manifest.update(new_stats)
        self._write_manifest(key, new_manifest)     # after the data file: a crash in between only causes a re-read

    def read(self, key: str, instruments) -> pa.Table:
        '''
        The stored rows of `instruments`, in stored order (by instrument, then original row order).
        Filter here instead of with `instrument IN (...)` in SQL: DuckDB turns a long IN list into a join, and the
        un-partitioned window used for rolling sums (loadData) then no longer sees the same row stream as a file scan.
        '''
        tbl = pq.read_table(self._path(key))
        return tbl.filter(pc.is_in(tbl['instrument'], value_set=pa.array(list(instruments), pa.string())))

    def discard(self, key):
        for p in (self._path(key), self._manifest_path(key)):
            try:
                os.remove(p)
            except OSError:
                pass

    def ensure(self, key: str, instruments):
        '''
        Bring the store up to date for `instruments`, re-reading only the stock files that are new or changed.
        Returns (store_path, instruments_with_rows_available) or None when the caller must scan the files itself.
        '''
        try:
            current = self._scan(key)
            wanted = sorted(i for i in instruments if i in current)
            if not wanted:
                return None

            path = self._path(key)
            manifest = self._load_manifest(key)
            if manifest and not self._readable(path):
                manifest = {}
            needed = [i for i in wanted if manifest.get(i) != current[i]]
            gone = [i for i in instruments if i in manifest and i not in current]

            if needed or gone:
                with self._lock(key) as locked:
                    if not locked:
                        return None
                    # another process may have brought it up to date while we waited
                    manifest = self._load_manifest(key)
                    if manifest and not self._readable(path):
                        manifest = {}
                    needed = [i for i in wanted if manifest.get(i) != current[i]]
                    gone = [i for i in instruments if i in manifest and i not in current]
                    if needed or gone:
                        new_tbl = self._read_files(key, needed) if needed else self._empty(key)
                        self._apply(key, manifest, new_tbl, {i: current[i] for i in needed}, gone)
            return path, wanted
        except Exception as e:
            print(f'[RawStore] {key}: falling back to scanning the stock files ({type(e).__name__}: {e})')
            return None

    def upsert(self, key: str, items: dict) -> bool:
        '''
        Ingestion hook. items = {instrument: (DataFrame just written to db/{instrument}_{key}.parquet, [mtime_ns, size])}
        Returns False if skipped; the next ensure() then repairs the store from the stock files.
        '''
        with self._lock(key) as locked:
            if not locked:
                return False
            manifest = self._load_manifest(key)
            if manifest and not self._readable(self._path(key)):
                manifest = {}

            frames = [df.reset_index() for df, _ in items.values() if df is not None and len(df)]
            if frames:
                new_tbl = self._normalize(pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False), key)
            else:
                new_tbl = self._empty(key)

            self._apply(key, manifest, new_tbl, {i: list(stat) for i, (_, stat) in items.items()}, set())
            return True
