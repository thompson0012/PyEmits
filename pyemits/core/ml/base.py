from pyemits.common.config_model import ConfigBase
from pyemits.common.validation import check_all_type_uniform
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.data_model import BaseDataModel
from typing import List
from abc import abstractmethod, ABC
import joblib


class TrainerBase(ABC):
    def __init__(self,
                 algo,
                 algo_config: List[ConfigBase],
                 ):
        self.algo = algo
        self.algo_config = algo_config
        self.meta_data_model = BaseDataModel()
        assert len(algo) == len(
            algo_config), f"length not matched, algo*{len(algo)} and algo_config*{len(algo_config)}  "

    @abstractmethod
    def _fit(self):
        pass

    def fit(self):
        return self._fit()


class PredictorBase(ABC):
    def __init__(self, clf_models):
        self.clf_models = clf_models

    @abstractmethod
    def _predict(self, data_model):
        pass

    def predict(self, data_model):
        return self._predict(data_model)


def save_model(clf_models, name):
    from pathlib import Path
    cwd = Path.cwd()
    print("current dir:", cwd)

    f = joblib.dump(clf_models, str(cwd.joinpath(name)) + '.pkl')
    print(f'model: {name}.pkl successfully saved')
    return f


def load_model(path_or_url):
    return joblib.load(path_or_url)
