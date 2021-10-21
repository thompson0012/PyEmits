"""
mean
median
freq
4分差
平均差
variance
sd

"""
from typing import Union, List
from modin import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from enum import Enum

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, \
    mean_squared_log_error

from pyemits.common.validation import raise_if_incorrect_type, raise_if_value_not_contains


class NumericalStats(Enum):
    @classmethod
    def available_list(cls):
        available_list = [attr for attr, obj in vars(NumericalStats).items() if isinstance(obj, classmethod)]
        available_list.remove('available_list')
        return available_list

    @classmethod
    def min(cls, series: Union[List, pd.Series]):
        return np.min(series)

    @classmethod
    def max(cls, series: Union[List, pd.Series]):
        return np.max(series)

    @classmethod
    def quantile(cls, series: Union[List, pd.Series],
                 quantile: float):
        return np.quantile(series, quantile)

    @classmethod
    def range(cls,
              minimum: Union[float, int],
              maximum: Union[float, int]):
        return maximum - minimum

    @classmethod
    def mean(cls, series: Union[List, pd.Series]):
        return np.mean(series)

    @classmethod
    def median(cls,
               series: Union[List, pd.Series]):
        return np.median(series)

    @classmethod
    def sd(cls,
           series: Union[List, pd.Series]):
        return np.std(series)

    @classmethod
    def sum(cls,
            series: Union[List, pd.Series]):
        return np.sum(series)

    @classmethod
    def skewness(cls,
                 series: Union[List, pd.Series, np.ndarray]):
        """

        Parameters
        ----------
        series : List, pd.Series, np.ndarray
            original data is expected, no need for histogram
        Returns
        -------

        """
        return skew(series)

    @classmethod
    def kurtosis(cls,
                 series: Union[List, pd.Series, np.ndarray]):
        """

        Parameters
        ----------
        series : List, pd.Series, np.ndarray
            original data is expected, no need for histogram
        Returns
        -------

        """
        return kurtosis(series)

    @classmethod
    def var(cls,
            series: Union[List, pd.Series, np.ndarray]):
        return np.var(series)

    @classmethod
    def null_counts(cls, series: pd.Series):
        return series.isnull().sum()

    @classmethod
    def null_pct(cls,
                 series: Union[pd.Series, pd.DataFrame]):
        return cls.null_counts(series) * 100 / len(series)

    @classmethod
    def inf_counts(cls,
                   series: Union[pd.Series, pd.DataFrame]):
        return series.isin({np.inf, -np.inf}).sum()

    @classmethod
    def inf_pct(cls,
                series: Union[pd.Series, pd.DataFrame]):
        return cls.inf_counts(series) * 100 / len(series)

    @classmethod
    def zeros_counts(cls,
                     series):
        return len(np.where(series == 0)[0])

    @classmethod
    def zeros_pct(cls,
                  series):
        return cls.zeros_counts(series) * 100 / len(series)

    @classmethod
    def neg_counts(cls,
                   series):
        return len(np.where(series < 0)[0])

    @classmethod
    def neg_pct(cls,
                series):
        return cls.neg_counts(series) * 100 / len(series)

    @classmethod
    def histogram(cls, series, bins, density=True):
        counts, edges = np.histogram(series, bins, density=density)
        return counts, edges


class RegressionMetrics(Enum):

    @classmethod
    def r2(cls, y_true, y_predicted):
        return np.round(r2_score(y_true, y_predicted), 2)

    @classmethod
    def mae(cls, y_true, y_predicted):
        return np.round(mean_absolute_error(y_true, y_predicted), 2)

    @classmethod
    def mape(cls, y_true, y_predicted):
        return np.round(mean_absolute_percentage_error(y_true, y_predicted), 2)

    @classmethod
    def rmse(cls, y_true, y_predicted):
        return np.round(np.sqrt(mean_squared_error(y_true, y_predicted)), 2)

    @classmethod
    def mse(cls, y_true, y_predicted):
        return np.round(mean_squared_error(y_true, y_predicted), 2)

    @classmethod
    def msle(cls, y_true, y_predicted):
        return np.round(mean_squared_log_error(y_true, y_predicted), 2)

    @classmethod
    def rmsle(cls, y_true, y_predicted):
        return np.round(np.sqrt(
            np.mean(np.power(np.log(np.array(abs(y_predicted)) + 1) - np.log(np.array(abs(y_true)) + 1), 2))), 2)

    @classmethod
    def full_list(cls):
        return ['r2', 'mae', 'mape', 'rmse', 'mse', 'msle']


def cal_reg_metrics(y_true: np.ndarray,
                    y_predicted: np.ndarray,
                    metrics: str or List[str] = 'all',
                    ):
    raise_if_incorrect_type(y_true, np.ndarray)
    raise_if_incorrect_type(y_predicted, np.ndarray)
    if isinstance(metrics, list):
        raise_if_value_not_contains(metrics, RegressionMetrics.full_list())

    elif isinstance(metrics, str):
        if metrics != 'all':
            raise ValueError('only accept "all" or List[str]')
        elif metrics == 'all':
            metrics = RegressionMetrics.full_list()

    metrics_result = {}
    for metric in metrics:
        metrics_result[metric] = getattr(RegressionMetrics, metric)(y_true, y_predicted)

    return metrics_result
