"""
data preview before data analysis and machine learning process
"""
from pyemits.common.stats import NumericalStats


class DataProfiler:
    def __init__(self):
        pass

    def summary(self):
        """
        columns count

        Returns
        -------

        """
        pass

    def stats(self):
        """
        min,
        max,
        quantile,
        range,
        mean,
        median,
        sd,
        sum,
        skewness,
        kurtosis,
        var,
        null_counts,
        null_pct,
        inf_counts,
        inf_pct,
        zeros_counts,
        zeros_pct,
        neg_counts,
        neg_pct,hist
        normal_dist_check
        qq-plot-r2

        var_type
        rows count
        distinct

        Returns
        -------

        """
        pass

    def viz(self):
        """
        corr plot
        hist
        qq-plot
        box-plot
        kde-plot
        category bar

        Returns
        -------

        """
        pass

    @classmethod
    def diff(cls, a, b):
        pass


