from pyemits.common.validation import raise_if_incorrect_type, raise_if_value_not_contains
from pyemits.common.errors import ItemNotFoundError
from typing import Union, List
import pandas as pd
import numpy as np
from pyemits.common.py_native_dtype import SliceableDeque


def list_ts_interval_counts(arr):
    arr = np.sort(arr)
    v_func = np.vectorize(lambda x: pd.Timedelta(x))
    time_diff = np.diff(arr)
    time_delta_arr = v_func(time_diff)
    unique, counts = np.unique(time_delta_arr, return_counts=True)
    return unique, counts


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
        if dataframe[i].dtype == np.dtype('datetime64[ns]'):
            return i

    raise ItemNotFoundError('timeindex colums not found in columns: ', list(dataframe.columns))


def extract_tensor_data(arr: Union[np.ndarray, pd.DataFrame],
                        window_size: int,
                        ravel=False):
    """

    Parameters
    ----------

    arr: np.ndarray | pd.DataFrame
        array like object, or DataFrame

    window_size: int
        window size, used in extracting the data in window basis

    ravel: bool, default = False
        flatten 2D array in 1D array

    Returns
    -------
    tensor_data: np.ndarray
        multi dimensional data, array

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df['time'] = pd.date_range('2021-01-01','2021-02-01',freq='15T')
    >>> df['values'] = range(0,len(df.index))
    >>> result = extract_tensor_data(df[['time','values']].values,5)
    >>> result
    ... array([[Timestamp('2021-01-01 00:00:00'), 0],
    ...   [Timestamp('2021-01-01 00:15:00'), 1],
    ...   [Timestamp('2021-01-01 00:30:00'), 2],
    ...   [Timestamp('2021-01-01 00:45:00'), 3],
    ...   [Timestamp('2021-01-01 01:00:00'), 4],
    ...   [Timestamp('2021-01-01 01:15:00'), 5],
    ...   [Timestamp('2021-01-01 01:30:00'), 6],
    ...   [Timestamp('2021-01-01 01:45:00'), 7],
    ...   [Timestamp('2021-01-01 02:00:00'), 8],
    ...   [Timestamp('2021-01-01 02:15:00'), 9],
    ...   [Timestamp('2021-01-01 02:30:00'), 10],
    ...   [Timestamp('2021-01-01 02:45:00'), 11],
    ...   [Timestamp('2021-01-01 03:00:00'), 12],
    ...   [Timestamp('2021-01-01 03:15:00'), 13],
    ...   [Timestamp('2021-01-01 03:30:00'), 14],
    ...   [Timestamp('2021-01-01 03:45:00'), 15],
    ...   [Timestamp('2021-01-01 04:00:00'), 16],
    ...   [Timestamp('2021-01-01 04:15:00'), 17],
    ...   [Timestamp('2021-01-01 04:30:00'), 18],
    ...   [Timestamp('2021-01-01 04:45:00'), 19]], dtype=object)
    """
    # hide, due to jit can't determine this
    # and function shall not do checking, shall move into function caller
    # raise_if_incorrect_type(window_size, int)

    length = len(arr)

    # use deque for 25%+ speed up
    # and allow slicing like list operation
    result = SliceableDeque()
    if ravel:
        for i in range(0, length):
            result.append(arr[i:i + window_size].ravel())

    else:
        for i in range(0, length):
            result.append(arr[i:i + window_size])

    return result[:-window_size]  # the last few [time steps] data will be in short


def make_future_datetime(last_date: str,
                         freq: Union[pd._libs.tslibs.timedeltas.Timedelta, int],
                         time_steps: int):
    """
    given specific last date, frequency, timesteps
    will generate the future datetime index for the use in forecasting

    Parameters
    ----------
    last_date: str
        it will be used as reference to generate future date

    freq: int | pd._libs.tslibs.timedeltas.Timedelta
        time interval for each steps

    time_steps: int
        time steps

    Returns
    -------
    datetime_idx: pd.DatetimeIndex

    Examples
    --------
    >>> pd.date_range('2021-01-01 00:00 +08:00', periods=5+1, freq='15T', closed='right')
    ... pd.DatetimeIndex(['2021-01-01 00:15:00+08:00', '2021-01-01 00:30:00+08:00',
    ...           '2021-01-01 00:45:00+08:00', '2021-01-01 01:00:00+08:00',
    ...           '2021-01-01 01:15:00+08:00'],
    ...          dtype='datetime64[ns, pytz.FixedOffset(480)]', freq='15T')

    """
    if isinstance(freq, int):
        freq = pd.Timedelta(freq)

    datetime_idx = pd.date_range(last_date, periods=time_steps + 1, freq=freq, closed='right')

    return datetime_idx


class SlidingWindowSplitter:
    def __init__(self, monitoring_windows: int, forecast_windows: int, ravel=False):
        """

        Parameters
        ----------
        monitoring_windows: int
            monitoring window size

        forecast_windows: int
            forecasting steps
        """
        self.monitoring_windows = monitoring_windows
        self.forecast_windows = forecast_windows
        self.ravel = ravel

        raise_if_incorrect_type(monitoring_windows, int)
        raise_if_incorrect_type(forecast_windows, int)

    def split(self, X, y=None):
        """

        Parameters
        ----------
        X: np.ndarray
            features

        y: np.ndarray, default = None
            target

        Returns
        -------

        """
        if y is not None:
            splitted_X = extract_tensor_data(X, window_size=self.monitoring_windows, ravel=self.ravel)[
                         :-self.forecast_windows]
            splitted_y = extract_tensor_data(y[self.monitoring_windows:], window_size=self.forecast_windows,
                                             ravel=self.ravel)
            return splitted_X, splitted_y

        splitted_X = extract_tensor_data(X, window_size=self.monitoring_windows, ravel=self.ravel)[:-self.forecast_windows]
        return splitted_X
