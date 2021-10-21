from abc import abstractmethod, ABC
from pyemits.common.validation import raise_if_incorrect_type
from typing import Union, Any, List
from pyemits.common.errors import ItemOverwriteError


class BaseDataModel(ABC):
    def __init__(self):
        self._meta_data = {}
        pass

    @property
    def meta_data(self):
        return self._meta_data

    def update_meta_data(self, kv):
        self._meta_data.update(kv)
        return

    def add_meta_data(self, k: str, v: Union[list, float, int, str, tuple]):
        raise_if_incorrect_type(k, str)
        raise_if_incorrect_type(k, (list, float, int, str, tuple))

        kv = self.meta_data.get(k, None)
        if kv is None:
            # create and instantly return
            self.update_meta_data({k: v})
            return

        elif isinstance(kv, list):
            # if passing a list, extend it
            # you can passing [[1,2,3]] to append list
            if isinstance(v, list):
                self._meta_data[k].extend(v)
                return

            # appending a new value
            self._meta_data[k].append(v)
            return

        elif isinstance(kv, (float, int, str, tuple, dict)):
            raise ItemOverwriteError(
                f"meta_data[{k}] is assigned as float/int/str/tuple/tuple, not not able to be overwrite in add method, "
                f"only list is allowed to add values"
                f"use update method to change values")

        raise TypeError

    def pop_meta_data(self, key: str):
        """
        provide marco-level removal of key elements, the whole elements of dict[key] will be removed.
        it is recommended to use method: "update_meta_data" if you want to make a micro-level changes on meta data

        Parameters
        ----------
        key: str

        Returns
        -------

        """
        self._meta_data.pop(key)
        return

    def _update_variable(self, name: str, values: Any):
        self.__register_update_variable(name, values)
        return

    def __register_update_variable(self, name: str, values: Any):
        """
        private method for loading data cls/config dict and add it into class's attributes

        Parameters
        ----------
        name: str
            variable name

        values: Any
            variable values

        Returns
        -------

        """
        raise_if_incorrect_type(name, str)
        self.__dict__.update({name: values})
        return


class ForecastDataModel(BaseDataModel):
    def __init__(self):
        super(ForecastDataModel, self).__init__()


class AnomalyDataModel(BaseDataModel):
    def __init__(self):
        super(AnomalyDataModel, self).__init__()


class EnsembleDataModel(BaseDataModel):
    def __init__(self):
        super(EnsembleDataModel, self).__init__()


class RegressionDataModel(BaseDataModel):
    def __init__(self, X_data, y_data=None):
        super(RegressionDataModel, self).__init__()
        self.X_data = X_data
        self.y_data = y_data


class KFoldCVDataModel(RegressionDataModel):
    def __init__(self, X_data, y_data):
        super(KFoldCVDataModel, self).__init__(X_data, y_data)
