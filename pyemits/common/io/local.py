import pandas as pd
from pyemits.common.typing.data_format import DataFormat
from typing import List
import httpx


class BaseLoader:
    def __init__(self, paths: str, file_format=None):
        self.paths = paths
        self.file_format = self._detect_file_format(file_format)
        self.file_loader = self._file_loader()

    def _detect_file_format(self, file_format):
        if file_format is not None:
            assert file_format in DataFormat.to_list()
            return file_format

        if self._is_url():
            return DataFormat.CSV

        file_path = self.paths[0]

        if file_path.endswith('.parquet'):
            return DataFormat.PARQUET

        elif file_path.endswith('.csv'):
            return DataFormat.CSV

        elif file_path.endswith('.json'):
            return DataFormat.JSON

    def _file_loader(self):
        if self.file_format == DataFormat.PARQUET:
            return pd.read_parquet

        elif self.file_format == DataFormat.CSV:
            return pd.read_csv

        elif self.file_format == DataFormat.JSON:
            return pd.read_json

        return None

    def _is_url(self):
        try:
            if httpx.get(self.paths[0]) == 200:
                return True
        except:
            return False


class ParallelLoader(BaseLoader):
    """
    ParallelLoader can't work with Modin, as Modin already do the Parallelization
    """
    def __init__(self, paths, file_format):
        super(ParallelLoader, self).__init__(paths, file_format)

    def load(self, concat=False, axis=1):
        def _load_df_from_path(path):
            loader = self.file_loader
            df = loader(path)
            return df

        from joblib import Parallel, delayed
        out = Parallel(n_jobs=-1)(delayed(_load_df_from_path)(path) for path in self.paths)
        if concat:
            return pd.concat(out, axis=axis)

        return out

