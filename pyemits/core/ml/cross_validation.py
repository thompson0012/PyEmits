import numpy as np
from sklearn.model_selection import KFold
from pyemits.common.config_model import KFoldConfig, asdict_data_cls, ConfigBase
from pyemits.common.data_model import KFoldCVDataModel,RegressionDataModel
from pyemits.common.py_native_dtype import SliceableDeque
from typing import Tuple, Union


class KFoldCV:
    """
    only split data into K-fold CV format, ml training will be handled by other module
    """

    def __init__(self, data_model: Union[KFoldCVDataModel, RegressionDataModel], kfold_config: Union[KFoldConfig,ConfigBase] = KFoldConfig()):
        self.data_model = data_model
        self.X, self.y = data_model.X_data, data_model.y_data
        self.kf_config = dict(kfold_config)
        self.kf = KFold(**self.kf_config)

    def split(self) -> SliceableDeque:
        result = SliceableDeque()
        for train_idx, test_idx in self.kf.split(self.X, self.y):
            result.append((train_idx, test_idx))
        return result
