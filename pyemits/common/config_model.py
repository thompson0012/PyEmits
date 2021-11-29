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


class BaseConfig(BaseModel):
    """
    root class for Config

    use dict(ConfigBase) can convert it to dictionary, no need to use asdict like dataclass
    Pydantic will check all the input type in runtime

    if you have trusted validated source, use .construct(**dict), which is 30x faster

    """
    pass


class TrainerOtherConfig(BaseConfig):
    """
    Trainer Other Config
    """
    pass


class RegressionTrainerOtherConfig(TrainerOtherConfig):
    fit_config: Optional[Union[Dict, BaseConfig]] = None
    kfold_config: Optional[Union[Dict, BaseConfig]] = None


class AnomalyTrainerOtherConfig(TrainerOtherConfig):
    pass


class PredictorOtherConfig(BaseConfig):
    """
    Predictor Other Config
    """
    pass


class RegressionPredictorOtherConfig(PredictorOtherConfig):
    pass


class AnomalyPredictorOtherConfig(PredictorOtherConfig):
    combination_config: Optional[Any] = None  # see combo package for more details
    standard_scaler: Optional[Any] = None


class KFoldConfig(BaseConfig):
    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = None


class RFConfig(BaseConfig):
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


class GBDTConfig(BaseConfig):
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


class HGBDTConfig(BaseConfig):
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


class AdaBoostConfig(BaseConfig):
    base_estimator: Optional[Any] = None
    n_estimators: int = 50
    learning_rate: float = 1.0
    loss: Literal['linear', 'square', 'exponential'] = 'linear'
    random_state: Optional[int] = None


class MLPConfig(BaseConfig):
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


class ElasticNetConfig(BaseConfig):
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


class RidgeConfig(BaseConfig):
    alpha: Union[float, Any] = 1.0
    fit_intercept: bool = True
    normalize: bool = False
    copy_X: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-3
    solver: Literal['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'] = 'auto'
    positive: bool = False
    random_state: Optional[int] = None


class LassoConfig(BaseConfig):
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


class BayesianRidgeConfig(BaseConfig):
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


class HuberConfig(BaseConfig):
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


class XGBoostConfig(BaseConfig):
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


class XGBoostFitConfig(BaseConfig):
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


class LGBMConfig(BaseConfig):
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


class LGBMFitConfig(BaseConfig):
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


class KerasSequentialConfig(BaseConfig):
    layer: List[Any]
    compile: Dict[str, Any]


class TorchLightningSequentialConfig(BaseConfig):
    layer: List[Any]


class PyodAbodConfig(BaseConfig):
    contamination: float = 0.1
    n_neighbors: int = 5
    method: Literal['fast', 'default'] = 'fast'


class PyodAutoencoderConfig(BaseConfig):
    hidden_neurons: Optional[List] = [64, 32, 32, 64]
    hidden_activation: Optional[str] = 'relu'
    output_activation: Optional[str] = 'sigmoid'
    loss: Optional[Union[str, Any]] = 'mean_squared_error'
    optimizer: Optional[str] = 'adam'
    epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.2
    l2_regularizer: float = 0.1
    validation_size: float = 0.1
    preprocessing: bool = True
    verbose: int = 1
    random_state: Optional[int] = None
    contamination: float = 0.1


class PyodAutoendcoderTorchConfig(BaseConfig):
    hidden_neurons: Optional[List] = [64, 32]
    hidden_activation: Optional[str] = 'relu'
    batch_norm: Optional[bool] = True
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.2
    weight_decay: float = 1e-05
    preprocessing: bool = True
    loss_fn: Optional[Any] = None
    contamination: float = 0.1
    device: Optional[Any] = None


class PyodCblofConfig(BaseConfig):
    n_cluster: int = 8
    contamination: float = 0.1
    clustering_estimator: Optional[Any] = None
    alpha: float = 0.9
    beta: int = 5
    use_weights: bool = False
    check_estimator: bool = False
    random_state: Optional[int] = None
    n_jobs: int = 1


class PyodCofConfig(BaseConfig):
    contamination: float = 0.1
    n_neighbors: int = 20
    method: Literal['fast', 'memory'] = 'fast'


class PyodCombinationConfig(BaseConfig):
    """
    to be designed
    """


class PyodCopodConfig(BaseConfig):
    contamination: float = 0.1
    n_jobs: int = 1


class PyodDeepSvddConfig(BaseConfig):
    c: Optional[Union[float, str]] = None
    use_ae: Optional[bool] = False
    hidden_neurons: Optional[List] = [64, 32]
    hidden_activation: Optional[str] = 'relu'
    output_activation: Optional[str] = 'sigmoid'
    optimizer: Optional[str] = 'adam'
    epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.2
    l2_regularizer: float = 0.1
    validation_size: float = 0.1
    preprocessing: bool = True
    verbose: int = 1
    random_state: Optional[int] = None
    contamination: float = 0.1


class PyodFeatureBaggingConfig(BaseConfig):
    """
    to be designed
    """


class PyodHbosConfig(BaseConfig):
    n_bins: int = 10
    alpha: float = 0.1
    tol: float = 0.5
    contamination: float = 0.1


class PyodIforestConfig(BaseConfig):
    n_estimators: int = 100
    max_samples: Union[int, float, str] = 'auto'
    contamination: float = 0.1
    max_features: Union[int, float] = 1.0
    bootstrap: bool = False
    n_jobs: int = 1
    behaviour: str = 'old'
    random_state: Optional[int] = None
    verbose: int = 0


class PyodKnnConfig(BaseConfig):
    contamination: float = 0.1
    n_neighbors: int = 5
    method: Literal['largest', 'mean', 'median'] = 'largest'
    radius: float = 1.0
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: int = 30
    metric: Union[str, Callable] = 'minkowski'
    p: int = 2
    metric_params: Optional[Dict] = None
    n_jobs: int = 1


class PyodLmddConfig(BaseConfig):
    """
    to be filled
    """


class PyodLodaConfig(BaseConfig):
    """
    to be filled
    """


class PyodLofConfig(BaseConfig):
    n_neighbors: int = 20
    algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = 'auto'
    leaf_size: int = 30
    metric: Union[Callable, str] = 'minkowski'
    p: int = 2
    metric_params: Optional[Dict] = None
    contamination: float = 0.1
    n_jobs: int = 1
    novelty: bool = True


class PyodLociConfig(BaseConfig):
    """
    to be filled
    """


class PyodLscpConfig(BaseConfig):
    """
    to be filled
    """


class PyodMadConfig(BaseConfig):
    """
    to be filled
    """


class PyodMcdConfig(BaseConfig):
    """
    to be filled
    """


class PyodMoGaalConfig(BaseConfig):
    """
    to be filled
    """


class PyodOcsvmConfig(BaseConfig):
    """
    to be filled
    """


class PyodPcaConfig(BaseConfig):
    n_components: Optional[Union[int, float, str]] = None
    n_selected_components: Optional[int] = None
    contamination: float = 0.1
    # copy: bool = True # Pydantic attribute, not able to use this variables
    whiten: bool = False
    svd_solver: Literal['auto', 'full', 'arpack', 'randomized'] = 'auto'
    tol: float = 0.0
    iterated_power: Union[int, str] = 'auto'
    random_state: Optional[int] = None
    weighted: bool = True
    standardization: bool = True


class PyodRodConfig(BaseConfig):
    """
        to be filled
    """


class PyodSodConfig(BaseConfig):
    """
        to be filled
    """


class PyodSoGaalConfig(BaseConfig):
    """
        to be filled
    """


class PyodSosConfig(BaseConfig):
    """
        to be filled
    """


class PyodSuodConfig(BaseConfig):
    """
        to be filled
    """


class PyodVaeConfig(BaseConfig):
    """
        to be filled
    """


class PyodXgbodConfig(BaseConfig):
    """
        to be filled
    """
