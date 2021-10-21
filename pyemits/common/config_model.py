"""
store all config model that used in class/function arguments
this module will utilize the python native lib
see details in: https://docs.python.org/3/library/dataclasses.html
"""

from dataclasses import dataclass, asdict
from pyemits.common.validation import raise_if_not_dataclass
from typing import Literal, Optional, Union, List, Callable, Any, Tuple, Dict
from pydantic import BaseModel
import numpy as np


def asdict_data_cls(data_cls):
    raise_if_not_dataclass(data_cls)
    return asdict(data_cls)


glob_random_state = 0


class ConfigBase(BaseModel):
    """
    root class for Config

    use dict(ConfigBase) can convert it to dictionary, no need to use asdict like dataclass
    Pydantic will check all the input type in runtime

    if you have trusted validated source, use .construct(**dict), which is 30x faster

    """
    pass


class KFoldConfig(ConfigBase):
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = None


class RFConfig(ConfigBase):
    criterion: Literal['squared_error', 'absolute_error', 'poisson'] = 'squared_error'
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[Literal['auto', 'sqrt', 'log2'], int, float] = 'auto'
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = None
    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0  # non -negative float
    max_samples: Optional[Union[int, float]] = None


class GBDTConfig(ConfigBase):
    loss: Literal['squared_error', 'ls', 'absolute_error', 'lad', 'huber', 'quantile'] = 'squared_error'
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Literal['friedman_mse', 'squared_error'] = 'friedman_mse'
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    init: Optional[str] = None  # estimator or 'zero'
    random_state: Optional[int] = None
    max_features: Optional[Union[Literal['auto', 'sqrt', 'log2'], int, float]] = None
    alpha: float = 0.9
    verbose: int = 0
    max_leaf_nodes: Optional[int] = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: Optional[int] = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0  # non negative float


class HGBDTConfig(ConfigBase):
    """
    experimental features from sklearn
    """
    loss: Literal['squared_error', 'absolute_error', 'poisson'] = 'squared_error'
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: Optional[int] = 31
    max_depth: Optional[int] = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0
    max_bins: int = 255
    categorical_features: Optional[Union[List[bool], List[int]]] = None
    monotonic_cst: Optional[List[int]] = None
    warm_start: bool = False
    early_stopping: Union[Literal['auto'], bool] = 'auto'
    scoring: Optional[Union[str, Callable]] = 'loss'
    validation_fraction: Optional[Union[int, float]] = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    verbose: int = 0
    random_state: Optional[int] = None


class AdaBoostConfig(ConfigBase):
    base_estimator: Optional[Any] = None
    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: Literal['linear', 'square', 'exponential'] = 'linear'
    random_state: Optional[int] = None


class MLPConfig(ConfigBase):
    hidden_layer_sizes: tuple = (100,)
    activation: Literal['identity', 'logistic', 'tanh', 'relu'] = 'relu'
    solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam'
    alpha: float = 0.0001
    batch_size: Union[int, Literal['auto']] = 'auto'
    learning_rate: Literal['constant', 'invscaling', 'adaptive'] = 'constant'
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    max_iter: int = 200
    shuffle: bool = True
    random_state: Optional[int] = None
    tol: float = 1e-4
    verbose: bool = False
    warm_start: bool = False
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    max_fun: int = 15000


class ElasticNetConfig(ConfigBase):
    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    normalize: bool = False
    precompute: Union[bool, List, Any] = False
    max_iter: int = 100
    copy_X: bool = True
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: Optional[int] = None
    selection: Literal['cyclic', 'random'] = 'cyclic'


class RidgeConfig(ConfigBase):
    alpha: Union[float, Any] = 1.0
    fit_intercept: bool = True
    normalize: bool = False
    copy_X: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-3
    solver: Literal['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'] = 'auto'
    positive: bool = False
    random_state: Optional[int] = None


class LassoConfig(ConfigBase):
    alpha: float = 1.0
    fit_intercept: bool = True
    normalize: bool = False
    precompute: Union[Literal['auto'], bool, List] = False
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    random_state: Optional[int] = None
    selection: Literal['cyclic', 'random'] = 'cyclic'


class BayesianRidgeConfig(ConfigBase):
    n_iter: int = 300
    tol: float = 1e-3
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    alpha_init: Optional[float] = None
    lambda_init: Optional[float] = None
    compute_score: bool = False
    fit_intercept: bool = True
    normalize: bool = False
    copy_X: bool = True
    verbose: bool = False


class HuberConfig(ConfigBase):
    epsilon: float = 1.35  # >1.0
    max_iter: int = 100
    alpha: float = 0.0001
    warm_start: bool = False
    fit_intercept: bool = True
    tol: float = 1e-05


array_like = Any
Metric = Callable[[np.ndarray], Tuple[str, float]]
_SklObjective = Optional[
    Union[
        str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ]
]


class XGBoostConfig(ConfigBase):
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    n_estimators: int = 100
    verbosity: Optional[int] = None
    objective: _SklObjective = None
    booster: Optional[str] = None
    tree_method: Optional[str] = None
    n_jobs: Optional[int] = None
    gamma: Optional[float] = None
    min_child_weight: Optional[float] = None
    max_delta_step: Optional[float] = None
    subsample: Optional[float] = None
    colsample_bytree: Optional[float] = None
    colsample_bylevel: Optional[float] = None
    colsample_bynode: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    scale_pos_weight: Optional[float] = None
    base_score: Optional[float] = None
    random_state: Optional[int] = None
    missing: float = np.nan
    num_parallel_tree: Optional[int] = None
    monotone_constraints: Optional[Union[Dict[str, int], str]] = None
    interaction_constraints: Optional[Union[str, List[Tuple[str]]]] = None
    importance_type: Optional[str] = None
    gpu_id: Optional[int] = None
    validate_parameters: Optional[bool] = None
    predictor: Optional[str] = None
    enable_categorical: bool = False


class XGBoostFitConfig(ConfigBase):
    sample_weight: Optional[array_like] = None
    base_margin: Optional[array_like] = None
    eval_set: Optional[List[Tuple[array_like, array_like]]] = None
    eval_metric: Optional[Union[str, List[str], Metric]] = None
    early_stopping_rounds: Optional[int] = None
    verbose: Optional[bool] = True
    xgb_model: Optional[Union[str, "XGBModel"]] = None
    sample_weight_eval_set: Optional[List[array_like]] = None
    base_margin_eval_set: Optional[List[array_like]] = None
    feature_weights: Optional[array_like] = None
    callbacks: Optional[List] = None


class LGBMConfig(ConfigBase):
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample_for_bin: int = 200000
    objective: Optional[Union[str, Callable]] = None
    class_weight: Optional[Union[Dict, str]] = None
    min_split_gain: float = 0.
    min_child_weight: float = 1e-3
    min_child_samples: int = 20
    subsample: float = 1.
    subsample_freq: int = 0
    colsample_bytree: float = 1.
    reg_alpha: float = 0.
    reg_lambda: float = 0.
    random_state: Optional[Union[int]] = None
    n_jobs: int = -1
    silent: Union[bool, str] = 'warn'
    importance_type: str = 'split'


class LGBMFitConfig(ConfigBase):
    sample_weight: array_like = None
    init_score: array_like = None
    eval_set: Optional[List] = None
    eval_names: Optional[List[str]] = None
    eval_sample_weight: Optional[List[array_like]] = None
    eval_init_score: Optional[List[array_like]] = None
    eval_metric: Optional[Union[str, Callable, List]] = None
    early_stopping_rounds: Optional[int] = None
    verbose: Optional[Union[bool, int]] = True
    feature_name: Optional[Union[List[str], Literal['auto']]] = 'auto'
    categorical_features: Optional[Union[List[str], List[int], Literal['auto']]] = 'auto'
    callbacks: Optional[List[Callable]] = None
    init_model: Optional[Union[str, Any]] = None


class KerasSequentialConfig(ConfigBase):
    layer: List[Any]
    compile: Dict[str, Any]

