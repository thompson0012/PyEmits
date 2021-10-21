"""
Why you have to do feature scaling?
Feature scaling, commonly referred to as "feature normalization" and "standardization", is an important technology in data preprocessing, 
and sometimes even determines whether the algorithm can work and whether it works well. 
When it comes to the necessity of feature scaling, the two most commonly used examples may be:
 
1. The units (scales) between features may be different, such as height and weight, such as degrees Celsius and Fahrenheit,
such as the area of a house and the number of rooms, and the range of changes in a feature. It may be [1000, 10000], 
and the variation range of another feature may be [−0.1,0.2]. 
When calculating distances, different units will lead to different calculation results.
Large-scale features will play a decisive role. 
The function of small-scale features may be ignored.
In order to eliminate the influence of unit and scale differences between features, 
and treat each dimension feature equally, it is necessary to normalize the features. 
   
2. Under the original feature, due to the difference in scale, the contour map of the loss function may be elliptical, 
the gradient direction is perpendicular to the contour line, 
and the descending will follow the zigzag route instead of pointing to the local minimum. 
After performing the zero-mean and unit-variance transformation on the features, 
the contour map of the loss function is closer to a circle, the direction of the gradient descent is less oscillating,
and the convergence is faster. As shown in the figure below, the picture is from Andrew Ng. 


One thing that is confusing is to refer to confusion. Standardization refers to a clearer term, 
but normalization sometimes refers to min-max normalization, 
sometimes refers to Standardization, 
and sometimes refers to Scaling to unit length.
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from enum import Enum
from typing import Dict, Union, Literal
import numpy as np
from pyemits.common.validation import raise_if_value_not_between, raise_if_incorrect_type
from pyemits.core.preprocessing.misc_utils import zero_inf_nan_handler


class FeatureScaling(Enum):
    @classmethod
    def min_max(cls):
        return MinMaxScaler

    @classmethod
    def standard(cls):
        return StandardScaler

    @classmethod
    def normalize(cls):
        return Normalizer

    @classmethod
    def robust(cls):
        return RobustScaler


class ForecastArrayScaling:

    @classmethod
    def auto_regressive_scale(cls,
                              X: np.ndarray,
                              y: np.ndarray,
                              handle_zero_nan_inf=True):
        """
        scale all the forecast value compared to the first element of X, which means X[0]
        so that machine learning is applied for forecasting the scale compared to the first element of X

        only be used when it is autoregressive (single features)
        if interval is high frequent, may not perform well

        it is idea only, pls test it with your own data for performance

        References
        ----------
        https://peerj.com/preprints/3190.pdf
        Forecasting at Scale Sean J. Taylor∗ † Facebook, Menlo Park, California, United States sjt@fb.com and Benjamin Letham† Facebook, Menlo Park, California, United States bletham@fb.com


        Parameters
        ----------

        X: array

        y: array

        handle_zero_nan_inf: bool
            if True, use default method to handle all zeros, nan, inf values

        See Also
        --------
        zero_inf_nan_handler

        Returns
        -------

        """

        raise_if_incorrect_type(X, np.ndarray)
        raise_if_incorrect_type(y, np.ndarray)
        X_result = np.zeros(shape=X.shape)
        y_result = np.zeros(shape=y.shape)

        if handle_zero_nan_inf == 'default':
            for i in range(len(X)):
                X_result[i] = X[i] / zero_inf_nan_handler(X)[i][0]
                y_result[i] = y[i] / zero_inf_nan_handler(X)[i][0]
            return zero_inf_nan_handler(X_result), zero_inf_nan_handler(y_result)

        if isinstance(handle_zero_nan_inf, dict):
            for i in range(len(X)):
                X_result[i] = X[i] / zero_inf_nan_handler(X, **handle_zero_nan_inf)[i][0]
                y_result[i] = y[i] / zero_inf_nan_handler(X, **handle_zero_nan_inf)[i][0]
            return zero_inf_nan_handler(X_result, **handle_zero_nan_inf), zero_inf_nan_handler(y_result, **handle_zero_nan_inf)

        return X_result, y_result

    @classmethod
    def revert_auto_regressive_scale(cls,
                                     X: np.ndarray,
                                     scaled_prediction_y: np.ndarray,
                                     handle_zero_nan_inf: Union[Literal['default'], dict] = 'default'):
        raise_if_incorrect_type(X, np.ndarray)
        raise_if_incorrect_type(scaled_prediction_y, np.ndarray)

        first_element = np.array(X)[:, 0].reshape(-1, 1)

        if handle_zero_nan_inf == 'default':
            scaled_back_y = zero_inf_nan_handler(first_element) * zero_inf_nan_handler(scaled_prediction_y)
            return zero_inf_nan_handler(scaled_back_y)

        elif isinstance(handle_zero_nan_inf, dict):
            scaled_back_y = zero_inf_nan_handler(first_element, **handle_zero_nan_inf) * zero_inf_nan_handler(scaled_prediction_y, **handle_zero_nan_inf)
            return zero_inf_nan_handler(scaled_back_y)

        scaled_back_y = first_element * scaled_prediction_y
        return zero_inf_nan_handler(scaled_back_y)
