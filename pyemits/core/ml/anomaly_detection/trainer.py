from pyemits.core.ml.base import BaseTrainer, BaseWrapper
from pyemits.common.data_model import AnomalyDataModel
from pyemits.common.config_model import BaseConfig
from pyemits.common.data_model import MiscContainer
from pyemits.common.validation import raise_if_value_not_contains, raise_if_not_init_cls
from typing import List, Union, Optional, Dict
from pyemits.common.py_native_dtype import SliceableDeque

from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.base import BaseDetector

# more algorithm will be added to the container
# now includes common algorithm for fast development
AnomalyModelContainer = {'PCA': PCA,
                         'COF': COF,
                         'KNN': KNN,
                         'IForest': IForest,
                         'LOF': LOF,
                         # 'XGBOD': XGBOD,
                         'HBOS': HBOS,
                         }


class PyodWrapper(BaseWrapper):
    def __init__(self, pyod_model_obj: BaseDetector, nickname=None):
        super(PyodWrapper, self).__init__(pyod_model_obj, nickname=nickname)
        self._model_obj: BaseDetector

    @property
    def model_obj(self) -> BaseDetector:
        return self._model_obj

    @classmethod
    def from_blank_model(cls, non_init_model):
        return cls(non_init_model)

    def init_pyod_model(self, *args, **kwargs):
        from pyemits.common.validation import is_initialized_cls
        if not is_initialized_cls(self.model_obj):
            self._model_obj = self._model_obj(*args, **kwargs)
            return
        raise TypeError('already initialized obj, not a class')

    def _fit(self, *args, **kwargs):
        raise_if_not_init_cls(self.model_obj)
        return self.model_obj.fit(*args, **kwargs)

    def get_contamination(self):
        return self.model_obj.contamination

    def decision_scores(self):
        return self.model_obj.decision_scores_

    def decision_function(self, X):
        return self.model_obj.decision_function(X)

    def _predict(self, *args, **kwargs):
        raise_if_not_init_cls(self.model_obj)
        return self.model_obj.predict(*args, **kwargs)

    def __str__(self):
        return f"PyodWrapper_{self._nickname}"


class AnomalyTrainer(BaseTrainer):
    """

    Parameters
    ----------
    algo_or_wrapper: List[Union[str, BaseWrapper]]

    algo_config: List[Optional[Union[BaseConfig, Dict]]]

    raw_data_model: AnomalyDataModel

    other_config: Dict
    """

    def __init__(self,
                 algo_or_wrapper: List[Union[str, BaseWrapper]],
                 algo_config: List[Optional[Union[BaseConfig, Dict]]],
                 raw_data_model: AnomalyDataModel,
                 other_config: Dict = {},
                 ):
        BaseTrainer.__init__(self, algo_or_wrapper, algo_config)
        # BasePredictor.__init__(self, clf_models=None)

        self._raw_data_model = raw_data_model
        self._other_config = other_config
        self._clf_models = SliceableDeque()
        self._is_algo_valid()
        self._is_algo_config_valid()

        self._misc_container = MiscContainer()

    @property
    def misc_container(self):
        return self._misc_container

    @property
    def clf_models(self):
        return self._clf_models

    @property
    def other_config(self):
        return self._other_config

    def _is_algo_valid(self):
        for item in self._algo:
            if not isinstance(item, (str, PyodWrapper)):
                raise TypeError('only accept str and PyodWrapper')

            if isinstance(item, str):
                raise_if_value_not_contains([item], list(AnomalyModelContainer.keys()))

            if isinstance(item, PyodWrapper):
                continue

    def _is_algo_config_valid(self):
        for item in self._algo_config:
            if item is None:
                continue

            if not isinstance(item, (BaseConfig, Dict)):
                # limit the parameters input
                raise TypeError('algo config only accept ConfigBase as input')

    @staticmethod
    def _get_detector(algo_or_wrapper: Union[str, PyodWrapper]):
        if isinstance(algo_or_wrapper, str):
            model_obj = AnomalyModelContainer[algo_or_wrapper]  # not activated
            return PyodWrapper(model_obj, algo_or_wrapper)

        elif isinstance(algo_or_wrapper, PyodWrapper):
            return algo_or_wrapper

        raise TypeError('only accept str or PyodWrapper')

    @staticmethod
    def _fit_algo_config(pyod_wrapper: PyodWrapper,
                         algo_config: Optional[Union[BaseConfig, Dict]]):
        if algo_config is not None:
            if isinstance(algo_config, BaseConfig):
                pyod_wrapper.init_pyod_model(**dict(algo_config))
                return pyod_wrapper

            pyod_wrapper.init_pyod_model(**algo_config)
            return pyod_wrapper

        elif algo_config is None:
            from pyemits.common.validation import is_initialized_cls
            if not is_initialized_cls(pyod_wrapper.model_obj):
                pyod_wrapper.init_pyod_model()
                return pyod_wrapper
            return pyod_wrapper

    def _fit(self):
        X = self._raw_data_model.X_data
        y = self._raw_data_model.y_data

        for i, (algo, algo_config) in enumerate(zip(self._algo, self._algo_config)):
            detector = self._get_detector(algo)
            detector = self._fit_algo_config(detector,
                                             algo_config)  # already a model object, support fit(), predict() method
            if y is None:
                detector.fit(X)
                self._clf_models.append((str(algo), detector))
            else:
                detector.fit(X, y)
                self._clf_models.append((str(algo), detector))

        return



