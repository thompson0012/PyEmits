from pyemits.common.data_model import MetaDataModel
from pyemits.common.typing import StrEnum
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.validation import raise_if_incorrect_type
from pandas import DataFrame
from numpy import ndarray
from typing import Callable, List
import numpy as np


class DataNodeMetaEnum(StrEnum):
    dtype: str
    dimension: tuple
    size: int


class DataNode:
    def __init__(self, data):
        self._data = data
        self._meta_data_model = MetaDataModel()
        self.register_data()

    @property
    def data(self):
        return self._data

    @classmethod
    def from_raw(cls, data):
        return cls(data)

    def register_data(self):
        self._meta_data_model.add_meta_data('dtype', type(self._data))
        self._add_data_according2type()

    def _add_data_according2type(self):
        if self._meta_data_model.get_meta_data('dtype') is DataFrame:
            self._data: DataFrame
            self._meta_data_model.add_meta_data('dimension', self._data.shape)
            self._meta_data_model.add_meta_data('size', self._data.size)

        elif self._meta_data_model.get_meta_data('dtype') is ndarray:
            self._data: ndarray
            self._meta_data_model.add_meta_data('dimension', self._data.shape)
            self._meta_data_model.add_meta_data('size', self._data.size)

        raise TypeError('dtype is not recognized')


class Task:
    def __init__(self, func):
        """
        basic unit in data pipeline

        Parameters
        ----------
        func
        """
        self._task_func = func

    @property
    def name(self):
        return

    @property
    def description(self):
        return

    def register_args(self, *args, **kwargs):
        from toolz.curried import partial
        self._task_func = partial(self._task_func, *args, **kwargs)

    def execute(self, target):
        return self._task_func(target)


class Step:
    def __init__(self, name: str, tasks: List[Task], description: str):
        self._name = name
        self._tasks = tasks
        self._description = description

    @property
    def name(self):
        return self._name

    @property
    def tasks(self):
        return self._tasks

    def tasks_count(self):
        return len(self.tasks)

    def register_task(self, task):
        self._tasks.append(task)
        return


class Pipeline:
    def __init__(self):
        self._pipeline = SliceableDeque()
        self._pipeline_res = np.zeros(np.shape(self._pipeline))

    def register_step(self, step: Step):
        self._pipeline.append(step)

    def transform(self, target_obj: DataNode):
        from copy import deepcopy
        res = deepcopy(target_obj)
        for i, step in enumerate(self._pipeline):
            for ii, task in enumerate(step):
                res = task.execute(res.data)
                self._pipeline_res[i, ii] = True
        return res
