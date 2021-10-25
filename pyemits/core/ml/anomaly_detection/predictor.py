from enum import Enum
from typing import Dict

import numpy as np
from pyod.models.combination import aom, moa, average, maximization

from pyemits.common.data_model import MiscContainer
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.validation import raise_if_value_not_contains, raise_if_values_not_same
from pyemits.core.ml.anomaly_detection.trainer import PyodWrapper
from pyemits.core.ml.base import BasePredictor


class AnomalyPredictor(BasePredictor):
    def __init__(self,
                 clf_models,
                 other_config: Dict = {},
                 combination_method='average'):
        super(AnomalyPredictor, self).__init__(clf_models)
        self._other_config = other_config
        self._combination_method = combination_method
        self._init_score_container()
        self._misc_container = MiscContainer()
        self._is_combination_method_valid()
        
    @property
    def anomaly_score(self):
        return self._anomaly_scores

    @property
    def misc_container(self):
        return self._misc_container

    @property
    def other_config(self):
        return self.other_config

    @property
    def combination_method(self):
        return self._combination_method

    def _init_score_container(self):
        self._anomaly_scores = SliceableDeque()

    def _is_combination_method_valid(self):
        raise_if_value_not_contains([self._combination_method], ModelCombination.available_methods())

    def _check_model_config(self):
        # some global setting must be same for model_combination purpose
        # check contamination
        contaimination = []
        for item in self._clf_models:
            wrapper_obj: PyodWrapper = item[1]
            contaimination.append(wrapper_obj.get_contamination())

        self._contamination = np.mean(contaimination)
        raise_if_values_not_same(contaimination)

    def _cal_anomaly_score(self, data_model):
        X = data_model.X_data
        self._init_score_container()
        self._check_model_config()
        combo_func = getattr(ModelCombination, self._combination_method)
        temp_scores_container = SliceableDeque()
        for model_container in self.clf_models:
            model: PyodWrapper = model_container[1]
            anomaly_scores: np.ndarray = model.decision_function(X)
            anomaly_scores = anomaly_scores.reshape(-1, 1)
            temp_scores_container.append(anomaly_scores)

        reshaped_data = np.concatenate(temp_scores_container, axis=1)

        # pyod author stated that scores have to be normalized before combination
        from sklearn.preprocessing import StandardScaler
        scaler = self._other_config.get('standard_scaler', None)
        if scaler is None:
            print('creating new standard scaler')
            scaler = StandardScaler()
            scaler.fit(reshaped_data)

        self._misc_container['standard_scaler'] = scaler
        assert isinstance(scaler, StandardScaler), "scaler is not a standard scaler object"
        scaled_data = scaler.transform(reshaped_data)

        config = self._other_config.get('combination_config', None)
        if config is None:
            config = {}
        else:
            if not isinstance(config, Dict):
                raise TypeError('combination_config must be Dict like')
        scores_after_comb = combo_func(scaled_data, **config)

        self._anomaly_scores.append(scores_after_comb)
        return self.anomaly_score

    def _predict(self, data_model):
        anomaly_score = self._cal_anomaly_score(data_model)
        threshold = np.quantile(anomaly_score, (1 - self._contamination))

        return (anomaly_score > threshold).astype(int).ravel()


class ModelCombination(Enum):
    @staticmethod
    def aom(arr, **kwargs):
        """
        average of maximization

        Parameters
        ----------
        arr
        kwargs

        Returns
        -------

        """
        return aom(arr, **kwargs)

    @staticmethod
    def moa(arr, **kwargs):
        """
        maximization of average

        Parameters
        ----------
        arr
        kwargs

        Returns
        -------

        """
        return moa(arr, **kwargs)

    @staticmethod
    def average(arr, **kwargs):
        """
        averaging

        Parameters
        ----------
        arr
        kwargs

        Returns
        -------

        """
        return average(arr, **kwargs)

    @staticmethod
    def maximization(arr, **kwargs):
        """maximization"""

        return maximization(arr, **kwargs)

    @staticmethod
    def available_methods():
        return ['aom', 'moa', 'average', 'maximization']