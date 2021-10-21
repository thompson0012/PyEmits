import numpy as np
import pandas as pd
from pyemits.common.validation import raise_if_incorrect_type, raise_if_value_not_contains
from pyemits.common.utils.misc_utils import slice_iterables


def zero_inf_nan_handler(arr, zero=1e-4, nan=1e-4, neginf=0, posinf=0):
    arr = np.where(arr == 0, zero, arr)
    return np.nan_to_num(arr, nan=nan, neginf=neginf, posinf=posinf)


def df_create_time_features(df: pd.DataFrame):
    if not (df.index.dtype != 'datetime64[ns]' or df.index.dtype != 'datetime64[ms]' or df.index.dtype != 'datetime64'):
        raise TypeError('index must be datetime index')

    df_copy = df.copy()
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['weekday'] = df_copy.index.weekday
    df_copy['hour'] = df_copy.index.hour
    df_copy['minutes'] = df_copy.index.minute
    return df_copy


def create_time_features(arr: np.ndarray, time_features='all'):
    raise_if_incorrect_type(arr, np.ndarray)
    if not (arr.dtype != 'datetime64[ns]' or arr.dtype != 'datetime64[ms]' or arr.dtype != 'datetime64'):
        raise TypeError('array type only accept datetime64')

    year_func = lambda x: str(x)[0:4]
    month_func = lambda x: str(x)[5:7]
    day_func = lambda x: str(x)[8:10]
    hour_func = lambda x: str(x)[11:13]
    minutes_func = lambda x: str(x)[14:16]

    light_weight_storage = {'year': year_func,
                            'month': month_func,
                            'day': day_func,
                            'hour': hour_func,
                            'minutes': minutes_func}

    container = []
    if time_features == 'all':
        for k in list(light_weight_storage.keys()):
            container.append(slice_iterables(light_weight_storage[k], arr, return_list=True))
        return container

    if isinstance(time_features, list):
        raise_if_value_not_contains(time_features, list(light_weight_storage.keys()))
        for feature in time_features:
            container.append(slice_iterables(light_weight_storage[feature], arr, return_list=True))

        return container

    raise ValueError('only accept List[str] or "all"')
