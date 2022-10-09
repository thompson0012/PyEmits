from pyemits.common.stats import NumericalStats
import pyemits.common.feature_engineering
from collections import defaultdict


class EDA:
    def __init__(self, data, config=None):
        self._data = data
        self._config = config


class SeriesEDA(EDA):
    def __init__(self, data, config=None):
        super(SeriesEDA, self).__init__(data, config)


class NumericalSeriesEDA(SeriesEDA):
    def __init__(self, data, config=None):
        super(NumericalSeriesEDA, self).__init__(data, config)
        self.check_load_config()

    def check_load_config(self):
        if self._config == None:
            self.load_default_config()
            return

        else:
            if 'eda_config' not in self._config.keys():
                raise KeyError('cant find eda config')
            if 'bins' not in self._config['eda_config'].keys():
                raise KeyError('cant find bins setting in eda_config')
            return

    def load_default_config(self):
        self._config = {'eda_config': {'bins': 10}}
        return

    def _create_eda(self):
        eda_config = self._config['eda_config']

        eda = defaultdict(dict)
        eda['min'] = min_ = NumericalStats.min(self._data)
        eda['max'] = max_ = NumericalStats.max(self._data)
        eda['quantile_05'] = NumericalStats.quantile(self._data, 0.05)
        eda['quantile_25'] = NumericalStats.quantile(self._data, 0.25)
        eda['quantile_75'] = NumericalStats.quantile(self._data, 0.75)
        eda['quantile_95'] = NumericalStats.quantile(self._data, 0.95)
        eda['IQR'] = (eda['quantile_75'] - eda['quantile_25'])
        eda['range'] = NumericalStats.range(min_, max_)
        eda['mean'] = NumericalStats.mean(self._data)
        eda['median'] = NumericalStats.median(self._data)
        eda['sd'] = NumericalStats.sd(self._data)
        eda['sum'] = NumericalStats.sum(self._data)
        eda['skewness'] = NumericalStats.skewness(self._data)
        eda['kurtosis'] = NumericalStats.kurtosis(self._data)
        eda['var'] = NumericalStats.var(self._data)
        eda['null_counts'] = NumericalStats.null_counts(self._data)
        eda['null_pct'] = NumericalStats.null_pct(self._data)
        eda['inf_counts'] = NumericalStats.inf_counts(self._data)
        eda['inf_pct'] = NumericalStats.inf_pct(self._data)
        eda['zero_counts'] = NumericalStats.zeros_counts(self._data)
        eda['zero_pct'] = NumericalStats.zeros_pct(self._data)
        eda['neg_counts'] = NumericalStats.neg_counts(self._data)
        eda['neg_pct'] = NumericalStats.neg_pct(self._data)
        eda['histogram'] = NumericalStats.histogram(self._data, eda_config['bins'])
        eda['unique'] = NumericalStats.unique(self._data)
        eda['distinct_counts'] = NumericalStats.distinct_counts(self._data)

        return eda

    def create_eda(self):

        return self._create_eda()


class CategoricalSeriesEDA(SeriesEDA):
    def __init__(self, data, config=None):
        super(CategoricalSeriesEDA, self).__init__(data, config)


class DatetimeSeriesEDA(SeriesEDA):
    def __init__(self, data, config=None):
        super(DatetimeSeriesEDA, self).__init__(data, config)


def detect_dtypes(data):
    raise NotImplementedError
