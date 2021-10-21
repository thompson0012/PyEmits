from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge, HuberRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pyemits.core.ml.base import TrainerBase
from pyemits.common.config_model import ConfigBase, KerasSequentialConfig
from pyemits.common.data_model import RegressionDataModel
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.validation import raise_if_value_not_contains
from typing import List, Dict, Optional, Union, Any
import numpy as np
from pyemits.core.ml.regression.nn import WrapperBase

RegModelContainer = {
    'RF': RandomForestRegressor,
    'GBDT': GradientBoostingRegressor,
    # 'HGBDT': HistGradientBoostingRegressor,
    'AdaBoost': AdaBoostRegressor,
    'MLP': MLPRegressor,
    'ElasticNet': ElasticNet,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'BayesianRidge': BayesianRidge,
    'Huber': HuberRegressor,
    'XGBoost': XGBRegressor,
    'LightGBM': LGBMRegressor
}


def _get_reg_model(algo_or_wrapper: Union[str, WrapperBase]):
    if isinstance(algo_or_wrapper, str):
        return RegModelContainer[algo_or_wrapper]
    # return wrapper model
    elif isinstance(algo_or_wrapper, WrapperBase):
        return algo_or_wrapper


class RegTrainer(TrainerBase):
    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[ConfigBase]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, ConfigBase, Any]] = {}):
        """

        Parameters
        ----------
        algo: List[str]
            the machine learning algorithm, any machine learning model that have fit/predict can used in here
        algo_config: List[ConfigBase] or List[None]
            the respective config model of algo
        raw_data_model: RegressionDataModel
            data model obj, stores data and meta data
        other_config: ConfigBase
            other global config, shall be used in its sub-class
        """
        super(RegTrainer, self).__init__(algo, algo_config)
        # raise_if_value_not_contains(algo, list(RegModelContainer.keys()))

        self.raw_data_model = raw_data_model
        self.other_config = other_config
        self.clf_models = SliceableDeque()

    def is_config_exists(self, config_key: str):
        config_item = self.other_config.get(config_key, None)
        if config_item is None:
            return False
        return True

    def get_fill_fit_config(self):
        fit_config = self.other_config.get('fit_config', None)
        if isinstance(fit_config, list):
            assert len(fit_config) == len(self.algo), 'length not matched'
            return fit_config
        elif fit_config is None:
            fit_config_ = []  # rename variable
            for i in range(len(self.algo)):
                fit_config_.append(None)
            fit_config = fit_config_  # pointer,
            return fit_config
        else:
            raise TypeError('fit config not a list type')

    def fill_algo_config_clf(self,
                             clf_or_wrapper,
                             algo_config: Optional[ConfigBase] = None):
        # nn wrapper
        if isinstance(clf_or_wrapper, WrapperBase):
            if algo_config is not None:
                if isinstance(algo_config, KerasSequentialConfig):
                    for i in algo_config.layer:
                        clf_or_wrapper.nn_model_obj.add(i)
                    clf_or_wrapper.nn_model_obj.compile(**algo_config.compile)
                    return clf_or_wrapper
                raise TypeError('now only support KerasSequentialConfig')

            return clf_or_wrapper

        # sklearn clf path
        if algo_config is None:
            return clf_or_wrapper()  # activate
        else:
            return clf_or_wrapper(**dict(algo_config))

    def fill_fit_config_clf(self,
                            clf_or_wrapper,
                            X,
                            y,
                            fit_config: Optional[Union[ConfigBase, Dict]] = None,
                            ):
        # nn wrapper
        if isinstance(clf_or_wrapper, WrapperBase):
            if fit_config is None:
                return clf_or_wrapper.nn_model_obj.fit(X, y)

            if isinstance(fit_config, ConfigBase):
                return clf_or_wrapper.nn_model_obj.fit(X, y, **dict(fit_config))

            elif isinstance(fit_config, Dict):
                return clf_or_wrapper.nn_model_obj.fit(X, y, **fit_config)

        # sklearn/xgboost/lightgbm clf
        else:
            if fit_config is None:
                return clf_or_wrapper.fit(X, y)

            else:
                assert isinstance(fit_config, ConfigBase), "fig_config type not matched"

                return clf_or_wrapper.fit(X, y, **dict(fit_config))

    def _fit(self):
        X = self.raw_data_model.X_data
        y = self.raw_data_model.y_data

        # make sure y is 1D array in RegTrainer

        fit_config = self.get_fill_fit_config()

        for n, (algo, algo_config) in enumerate(zip(self.algo, self.algo_config)):
            clf = self.fill_algo_config_clf(_get_reg_model(algo), algo_config)
            self.fill_fit_config_clf(clf, X, y, fit_config[n])
            self.clf_models.append((str(algo), clf))
        return


class ParallelRegTrainer(RegTrainer):
    def __init__(self,
                 algo: List[str],
                 algo_config: List[ConfigBase],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, ConfigBase, Any]] = {}):
        """
        handy function to realize parallel training

        Parameters
        ----------
        algo: List[str]
            the machine learning algorithm, any machine learning model that have fit/predict can used in here
        algo_config: List[ConfigBase] or List[None]
            the respective config model of algo
        raw_data_model: RegressionDataModel
            data model obj, stores data and meta data
        other_config: ConfigBase
            other global config, shall be used in its sub-class
        """
        super(ParallelRegTrainer, self).__init__(algo, algo_config, raw_data_model, other_config)

    def _fit(self):
        from joblib import Parallel, delayed
        parallel = Parallel(n_jobs=-1)

        def _get_fitted_trainer(algo: List,
                                algo_config: List[ConfigBase],
                                raw_data_model: RegressionDataModel,
                                other_config: Dict[str, ConfigBase] = {}):
            trainer = RegTrainer(algo, algo_config, raw_data_model, other_config)
            trainer.fit()  # fit config auto filled by RegTrainer, no need to handle
            return trainer

        out: List[RegTrainer] = parallel(
            delayed(_get_fitted_trainer)([algo_], [algo_config_], self.raw_data_model, self.other_config) for
            algo_, algo_config_ in
            zip(self.algo, self.algo_config))

        # self.clf_models = [obj.clf_models for obj in out]
        for obj in out:
            self.clf_models.append(obj.clf_models)
        return

    def fit(self):
        return self._fit()


class MultiOutputRegTrainer(RegTrainer):
    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[ConfigBase]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, ConfigBase, Any]] = {},
                 parallel_n_jobs: int = -1):
        super(MultiOutputRegTrainer, self).__init__(algo, algo_config, raw_data_model, other_config)
        self.parallel_n_jobs = parallel_n_jobs

    def _fit(self):
        fit_config = self.get_fill_fit_config()

        from sklearn.multioutput import MultiOutputRegressor
        X = self.raw_data_model.X_data
        y = self.raw_data_model.y_data

        for n, (algo, algo_config) in enumerate(zip(self.algo, self.algo_config)):
            clf = self.fill_algo_config_clf(_get_reg_model(algo), algo_config)  # clf already activated
            clf = MultiOutputRegressor(estimator=clf, n_jobs=self.parallel_n_jobs)
            self.fill_fit_config_clf(clf, X, y, fit_config[n])
            self.clf_models.append((str(algo), clf))
        return


class KFoldCVTrainer(RegTrainer):
    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[ConfigBase]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, ConfigBase, Any]] = {},
                 ):
        super(KFoldCVTrainer, self).__init__(algo, algo_config, raw_data_model, other_config)

    def _fit(self):
        from pyemits.core.ml.cross_validation import KFoldCV
        kfold_config = self.other_config.get('kfold_config', None)
        if kfold_config is not None:
            kfold_cv = KFoldCV(self.raw_data_model, kfold_config)
        else:
            kfold_cv = KFoldCV(self.raw_data_model)

        splitted_kfold = kfold_cv.split()
        for n, item in enumerate(splitted_kfold):
            self.meta_data_model.add_meta_data('kfold_record', [item])
            train_idx = item[0]
            test_idx = item[1]
            X_ = self.raw_data_model.X_data[train_idx]
            y_ = self.raw_data_model.y_data[train_idx]
            sliced_data_model = RegressionDataModel(X_, y_)
            trainer = ParallelRegTrainer(self.algo, self.algo_config, sliced_data_model,
                                         other_config=self.other_config)
            trainer.fit()
            self.clf_models.append((f'kfold_{n}', trainer.clf_models))
        return
