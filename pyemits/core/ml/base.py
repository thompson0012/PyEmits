from pyemits.common.config_model import ConfigBase
from pyemits.common.validation import check_all_type_uniform
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.data_model import MetaDataModel
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
        self.meta_data_model = MetaDataModel()
        self._make_blank_algo_config_if_None()
        assert len(algo) == len(
            self.algo_config), f"length not matched, algo*{len(algo)} and algo_config*{len(self.algo_config)}  "

    @abstractmethod
    def _is_algo_valid(self):
        pass

    @abstractmethod
    def _is_algo_config_valid(self):
        pass

    def _make_blank_algo_config_if_None(self):
        if self.algo_config is None:
            self.algo_config = [None] * len(self.algo)

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


class WrapperBase(ABC):
    def __init__(self, model_obj, nickname=None):
        self._model_obj = model_obj
        self._nickname = nickname

    @property
    def model_obj(self):
        return self._model_obj

    @abstractmethod
    def _fit(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self._fit(*args, **kwargs)

    @abstractmethod
    def _predict(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return self._predict(*args, **kwargs)


class NeuralNetworkWrapperBase(WrapperBase):
    def __init__(self, nn_model_obj, nickname=None):
        super(NeuralNetworkWrapperBase, self).__init__(nn_model_obj, nickname)
        self._nn_model_obj = nn_model_obj
        self._nickname = nickname

    @property
    def nn_model_obj(self):
        """
        equal to model obj

        """
        return self.model_obj
