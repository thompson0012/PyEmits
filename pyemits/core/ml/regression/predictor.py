from pyemits.core.ml.base import PredictorBase
from pyemits.common.data_model import RegressionDataModel
from pyemits.common.py_native_dtype import SliceableDeque
from typing import List
import numpy as np
from typing import Literal


class RegPredictor(PredictorBase):
    """
    universal regression predictor
    specify the train method, so it can detect the data structure and use appropriate prediction method
    """

    def __init__(self,
                 clf_models,
                 train_method: Literal['RegTrainer', 'ParallelRegTrainer', 'MultiOutputRegTrainer', 'KFoldCVTrainer']):
        super(RegPredictor, self).__init__(clf_models)
        self.clf_models_predictions = SliceableDeque()
        self.train_method = train_method

    def _get_clf_models_shape_dim(self):
        shape = np.shape(self.clf_models)
        return len(shape)

    def _init_clf_models_predictions(self):
        self.clf_models_predictions = SliceableDeque()

    def _predict(self, data_model: RegressionDataModel):
        X = data_model.X_data

        self._init_clf_models_predictions()

        if self.train_method == 'ParallelRegTrainer':
            for item in self.clf_models:
                algo = item[0][0]
                clf_obj = item[0][1]
                prediction_result = clf_obj.predict(X)

                # to match the original shape
                # Deque also for performance
                result_container = SliceableDeque()
                result_container.append(prediction_result)

                self.clf_models_predictions.append(result_container)

        # AST can't recognize syntax like: a == (1 or 2)
        elif self.train_method == 'RegTrainer' or self.train_method == 'MultiOutputRegTrainer':
            for item in self.clf_models:
                algo = item[0]
                clf_obj = item[1]
                prediction_result = clf_obj.predict(X)
                self.clf_models_predictions.append(prediction_result)

        elif self.train_method == 'KFoldCVTrainer':
            for each_kfold_container in self.clf_models:
                n_kfold = each_kfold_container[0]
                model_containers = each_kfold_container[1]

                result_containers = SliceableDeque()
                for container in model_containers:
                    algo = container[0][0]
                    clf_obj = container[0][1]
                    prediction_result = clf_obj.predict(X)
                    result_containers.append(prediction_result)

                self.clf_models_predictions.append(result_containers)
        else:
            raise ValueError(f'train_method {self.train_method} is under development')

        return self.clf_models_predictions

# TODO - plan to add ParallelRegPredictor to speed up
