from pyemits.core.ml.base import TrainerBase, WrapperBase
from pyemits.common.data_model import AnomalyDataModel
from pyemits.common.config_model import ConfigBase
from pyemits.common.validation import raise_if_value_not_contains, raise_if_not_init_cls
from typing import List, Any, Union, Optional, Dict
from pyemits.common.py_native_dtype import SliceableDeque

from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.xgbod import XGBOD
from pyod.models.hbos import HBOS

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


class PyodWrapper(WrapperBase):
    def __init__(self, pyod_model_obj, nickname=None):
        super(PyodWrapper, self).__init__(pyod_model_obj, nickname=nickname)

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

    def _predict(self, *args, **kwargs):
        raise_if_not_init_cls(self.model_obj)
        return self.model_obj.predict(*args, **kwargs)

    def __str__(self):
        return f"PyodWrapper_{self._nickname}"


class AnomalyDetector(TrainerBase):
    def __init__(self,
                 algo_or_wrapper: List[Union[str, WrapperBase]],
                 algo_config: List[Optional[Union[ConfigBase, Dict]]],
                 raw_data_model: AnomalyDataModel,
                 other_config: Dict = {}):
        super(AnomalyDetector, self).__init__(algo_or_wrapper, algo_config)
        self._raw_data_model = raw_data_model
        self._is_algo_valid()
        self._is_algo_config_valid()
        self._detectors_container = SliceableDeque()

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

            if not isinstance(item, (ConfigBase, Dict)):
                # limit the parameters input
                raise TypeError('algo config only accept ConfigBase as input')

    @property
    def detectors(self):
        return self._detectors_container

    @staticmethod
    def _get_detector(algo: Union[str, PyodWrapper]):
        if isinstance(algo, str):
            model_obj = AnomalyModelContainer[algo]  # not activated
            return PyodWrapper(model_obj, algo)

        elif isinstance(algo, PyodWrapper):
            return PyodWrapper

        raise TypeError('only accept str or PyodWrapper')

    @staticmethod
    def _fit_algo_config(pyod_wrapper, algo_config: Optional[Union[ConfigBase, Dict]]):
        if algo_config is not None:
            if isinstance(algo_config, ConfigBase):
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
                self._detectors_container.append((str(algo), detector))
            else:
                detector.fit(X, y)
                self._detectors_container.append((str(algo), detector))

        return
