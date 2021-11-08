import numpy as np
import pandas as pd

from pyemits.common.errors import ItemNotFoundError
from pyemits.common.validation import raise_if_incorrect_type, raise_if_value_not_contains
from pyemits.common.utils.misc_utils import slice_iterables


def zero_inf_nan_handler(arr, zero=1e-4, nan=1e-4, neginf=0, posinf=0):
    arr = np.where(arr == 0, zero, arr)
    return np.nan_to_num(arr, nan=nan, neginf=neginf, posinf=posinf)


def df_create_time_features(df: pd.DataFrame):
    if df.index.dtype not in ('datetime64[ns]', 'datetime64[ms]', 'datetime64'):
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
    if arr.dtype not in ('datetime64[ns]', 'datetime64[ms]', 'datetime64'):
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


def get_freq_time_interval(arr):
    unique, counts = list_ts_interval_counts(arr)
    max_idx = np.argmax(counts)
    return unique[max_idx]


def locate_timeindex_columns(dataframe):
    """
    locate the series data which is timeindex
    only return the first values

    Parameters
    ----------
    dataframe: pd.DataFrame

    Returns
    -------

    """
    for i in dataframe:
        # if dataframe[i].dtype == np.dtype('datetime64[ns]'):
        if dataframe[i].dtype in ('datetime64', 'datetime64[ms]', 'datetime64[ns]'):
            return i

    raise ItemNotFoundError('timeindex colums not found in columns: ', list(dataframe.columns))


def list_ts_interval_counts(arr):
    arr = np.sort(arr)
    v_func = np.vectorize(lambda x: pd.Timedelta(x))
    time_diff = np.diff(arr)
    time_delta_arr = v_func(time_diff)
    unique, counts = np.unique(time_delta_arr, return_counts=True)
    return unique, counts


def fix_async_freq(df: pd.DataFrame):
    df_ = df.copy()
    ts_idx = locate_timeindex_columns(df_)
    df_.set_index(ts_idx, inplace=True)

    arr = df_.sort_index().index.to_numpy()
    time_delta = get_freq_time_interval(arr)

    return df_.reindex(pd.date_range(arr[0], arr[-1], freq=time_delta))


def reduce_memory_usage(df, verbose=True):
    """
    References
    ----------
    https://mp.weixin.qq.com/s/RkiFbnGios5eDLiXftBGAw

    Parameters
    ----------
    df
    verbose

    Returns
    -------

    """
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
