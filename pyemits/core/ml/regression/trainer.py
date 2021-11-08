from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge, HuberRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pyemits.core.ml.base import BaseTrainer, BaseWrapper, NeuralNetworkWrapperBase
from pyemits.common.config_model import BaseConfig, KerasSequentialConfig, TorchLightningSequentialConfig
from pyemits.common.data_model import RegressionDataModel
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.validation import raise_if_value_not_contains
from typing import List, Dict, Optional, Union, Any

from pyemits.core.ml.regression.nn import TorchLightningWrapper

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


def _get_reg_model(algo_or_wrapper: Union[str, BaseWrapper]):
    if isinstance(algo_or_wrapper, str):
        return RegModelContainer[algo_or_wrapper]
    # return wrapper model
    elif isinstance(algo_or_wrapper, BaseWrapper):
        return algo_or_wrapper


def fill_algo_config_clf(clf_or_wrapper,
                         algo_config: Optional[BaseConfig] = None):
    # nn wrapper
    if isinstance(clf_or_wrapper, NeuralNetworkWrapperBase):
        # have algo config
        if algo_config is not None:
            # if keras model object
            if isinstance(algo_config, KerasSequentialConfig):
                for i in algo_config.layer:
                    clf_or_wrapper.model_obj.add(i)
                clf_or_wrapper.model_obj.compile(**algo_config.compile)
                return clf_or_wrapper
            elif isinstance(algo_config, TorchLightningSequentialConfig):
                clf_or_wrapper: TorchLightningWrapper
                for nos, layer in enumerate(algo_config.layer, 1):
                    clf_or_wrapper.add_layer2blank_model(str(nos), layer)
                return clf_or_wrapper
            # not support pytorch, mxnet model right now
            raise TypeError('now only support KerasSequentialConfig')

        # no algo config
        return clf_or_wrapper

    # sklearn clf path
    if algo_config is None:
        return clf_or_wrapper()  # activate
    else:
        return clf_or_wrapper(**dict(algo_config))


def fill_fit_config_clf(clf_or_wrapper,
                        X,
                        y,
                        fit_config: Optional[Union[BaseConfig, Dict]] = None,
                        ):
    from pyemits.core.ml.regression.nn import torchlighting_data_helper
    # nn wrapper
    if isinstance(clf_or_wrapper, NeuralNetworkWrapperBase):
        dl_train, dl_val = torchlighting_data_helper(X, y)

        if fit_config is None:
            # pytorch_lightning path
            if isinstance(clf_or_wrapper, TorchLightningWrapper):
                return clf_or_wrapper.fit(dl_train, dl_val)
            # keras path
            return clf_or_wrapper.fit(X, y)

        if isinstance(fit_config, BaseConfig):
            if isinstance(clf_or_wrapper, TorchLightningWrapper):
                return clf_or_wrapper.fit(dl_train, dl_val, **dict(fit_config))
            # keras path
            return clf_or_wrapper.fit(X, y, **dict(fit_config))

        elif isinstance(fit_config, Dict):
            if isinstance(clf_or_wrapper, TorchLightningWrapper):
                return clf_or_wrapper.fit(dl_train, dl_val, **fit_config)
            # keras path
            return clf_or_wrapper.fit(X, y, **fit_config)

    # sklearn/xgboost/lightgbm clf
    else:
        if fit_config is None:
            return clf_or_wrapper.fit(X, y)

        else:
            assert isinstance(fit_config, BaseConfig), "fig_config type not matched"

            return clf_or_wrapper.fit(X, y, **dict(fit_config))


class RegTrainer(BaseTrainer):
    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[BaseConfig]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, BaseConfig, Any]] = {}):
        """
        universal class for regression model training,
        all-in-one training including sklearn, xgboost, lightgbm, keras, pytorch_lightning

        you are not required to fill the algo config if you have idea on algo_config
        the algo config is designed for people to config their model based on the configuration that provided in config_model
        so that people can easily config their model during creation

        for Pytorch_lightning user, pls configured your model before use this. at that moment, no algo_config is

        Parameters
        ----------
        algo: List[str]
            the machine learning algorithm, any machine learning model that have fit/predict can used in here
        algo_config: List[BaseConfig] or List[None]
            the respective config model of algo
        raw_data_model: RegressionDataModel
            data model obj, stores data and meta data
        other_config: BaseConfig
            other global config, shall be used in its sub-class
        """
        super(RegTrainer, self).__init__(algo, algo_config)
        # raise_if_value_not_contains(algo, list(RegModelContainer.keys()))

        self.raw_data_model = raw_data_model
        self.other_config = other_config
        self.clf_models = SliceableDeque()
        self._is_algo_valid()
        self._is_algo_config_valid()

    def _is_algo_valid(self):
        for item in self._algo:
            if not isinstance(item, (str, NeuralNetworkWrapperBase)):
                raise TypeError('must be str or WrapperBase')
            if isinstance(item, str):
                raise_if_value_not_contains([item], list(RegModelContainer.keys()))

    def _is_algo_config_valid(self):
        for item in self._algo_config:
            if item is None:
                continue  # skip to next loop
            if not isinstance(item, (BaseConfig, Dict)):
                raise TypeError('Only accept ConfigBase or Dict as input')
            # no checking when model is object, which directly passing it

    def is_config_exists(self, config_key: str):
        config_item = self.other_config.get(config_key, None)
        if config_item is None:
            return False
        return True

    def get_fill_fit_config(self):
        fit_config = self.other_config.get('fit_config', None)
        if isinstance(fit_config, list):
            assert len(fit_config) == len(self._algo), 'length not matched'
            return fit_config
        elif fit_config is None:
            fit_config_ = []  # rename variable
            for i in range(len(self._algo)):
                fit_config_.append(None)
            fit_config = fit_config_  # pointer,
            return fit_config
        else:
            raise TypeError('fit config not a list type')

    def _fit(self):
        X = self.raw_data_model.X_data
        y = self.raw_data_model.y_data

        # make sure y is 1D array in RegTrainer

        fit_config = self.get_fill_fit_config()

        for n, (algo, algo_config) in enumerate(zip(self._algo, self._algo_config)):
            clf = fill_algo_config_clf(_get_reg_model(algo), algo_config)
            fill_fit_config_clf(clf, X, y, fit_config[n])
            self.clf_models.append((str(algo), clf))
        return


class ParallelRegTrainer(RegTrainer):
    def __init__(self,
                 algo: List[str],
                 algo_config: List[BaseConfig],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, BaseConfig, Any]] = {}):
        """
        handy function to realize parallel training

        Parameters
        ----------
        algo: List[str]
            the machine learning algorithm, any machine learning model that have fit/predict can used in here
        algo_config: List[BaseConfig] or List[None]
            the respective config model of algo
        raw_data_model: RegressionDataModel
            data model obj, stores data and meta data
        other_config: BaseConfig
            other global config, shall be used in its sub-class
        """
        super(ParallelRegTrainer, self).__init__(algo, algo_config, raw_data_model, other_config)

    def _fit(self):
        from joblib import Parallel, delayed
        parallel = Parallel(n_jobs=-1)

        def _get_fitted_trainer(algo: List,
                                algo_config: List[BaseConfig],
                                raw_data_model: RegressionDataModel,
                                other_config: Dict[str, BaseConfig] = {}):
            trainer = RegTrainer(algo, algo_config, raw_data_model, other_config)
            trainer.fit()  # fit config auto filled by RegTrainer, no need to handle
            return trainer

        out: List[RegTrainer] = parallel(
            delayed(_get_fitted_trainer)([algo_], [algo_config_], self.raw_data_model, self.other_config) for
            algo_, algo_config_ in
            zip(self._algo, self._algo_config))

        for obj in out:
            self.clf_models.append(obj.clf_models)
        return

    def fit(self):
        return self._fit()


class MultiOutputRegTrainer(RegTrainer):
    """
    machine learning based multioutput regression trainer
    bring forecasting power into machine learning model,
    forecasting is not only the power of deep learning
    """

    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[BaseConfig]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, BaseConfig, Any]] = {},
                 parallel_n_jobs: int = -1):
        super(MultiOutputRegTrainer, self).__init__(algo, algo_config, raw_data_model, other_config)
        self.parallel_n_jobs = parallel_n_jobs

    def _fit(self):
        fit_config = self.get_fill_fit_config()

        from sklearn.multioutput import MultiOutputRegressor
        X = self.raw_data_model.X_data
        y = self.raw_data_model.y_data

        for n, (algo, algo_config) in enumerate(zip(self._algo, self._algo_config)):
            clf = fill_algo_config_clf(_get_reg_model(algo), algo_config)  # clf already activated
            clf = MultiOutputRegressor(estimator=clf, n_jobs=self.parallel_n_jobs)
            fill_fit_config_clf(clf, X, y, fit_config[n])
            self.clf_models.append((str(algo), clf))
        return


class KFoldCVTrainer(RegTrainer):
    def __init__(self,
                 algo: List[Union[str, Any]],
                 algo_config: List[Optional[BaseConfig]],
                 raw_data_model: RegressionDataModel,
                 other_config: Dict[str, Union[List, BaseConfig, Any]] = {},
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
            self._meta_data_model.add_meta_data('kfold_record', [item])
            train_idx = item[0]
            test_idx = item[1]
            X_ = self.raw_data_model.X_data[train_idx]
            y_ = self.raw_data_model.y_data[train_idx]
            sliced_data_model = RegressionDataModel(X_, y_)
            trainer = ParallelRegTrainer(self._algo, self._algo_config, sliced_data_model,
                                         other_config=self.other_config)
            trainer.fit()
            self.clf_models.append((f'kfold_{n}', trainer.clf_models))
        return
