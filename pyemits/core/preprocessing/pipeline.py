from abc import abstractmethod
from typing import Callable, List, Iterable, Dict

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from pyemits.common.data_model import MetaDataModel
from pyemits.common.py_native_dtype import SliceableDeque
from pyemits.common.typing import StrEnum
from pyemits.common.validation import raise_if_incorrect_type, raise_if_not_all_type_uniform


class DataNodeMetaEnum(StrEnum):
    dtype: str
    dimension: tuple
    size: int


def _raise_if_value_not_equal(input_, expected):
    if input_ != expected:
        raise ValueError('value is not expected')
    return True


class DataNode:
    """
    the base of data node
    data node work as the default constructor
    help to create defined/desired meta data to fit users purpose on their domain

    you are required to rewrite the method: ['_add_data_according2type', 'check_input_data_type'] for inherit

    Parameters
    ----------
    data: Any
        raw input data source
    """

    def __init__(self, data):
        self._data = data
        self._meta_data_model = MetaDataModel()
        self._register_data()
        self._check_input_data_type()

    @property
    def data(self):
        return self._data

    @abstractmethod
    def _add_data_according2type(self):
        pass

    def _register_data(self):
        self._meta_data_model.add_meta_data('dtype', type(self._data))
        self._add_data_according2type()

    @property
    def meta_data(self):
        return self._meta_data_model.meta_data

    @abstractmethod
    def _check_input_data_type(self):
        pass


class DataNodeInferringEngine:
    def __init__(self, data_node_mapper=None):
        self._init_data_node_mapper(data_node_mapper)

    def _init_data_node_mapper(self, data_node_mapper):
        if not data_node_mapper:
            self._mapper = dict(zip([np.ndarray, pd.DataFrame, pd.Series],
                                    [NumpyDataNode, PandasDataFrameDataNode, PandasSeriesDataNode]))
        else:
            self._mapper = data_node_mapper

        return

    def infer(self, data):
        data_type = type(data)
        try:
            return self._mapper[data_type](data)
        except KeyError:
            raise KeyError('the input data type is not exist in the data node mapper')


def infer_DataNode_type(data, engine: DataNodeInferringEngine = None):
    if not engine:
        engine = DataNodeInferringEngine()

    raise_if_incorrect_type(engine, DataNodeInferringEngine)
    return engine.infer(data)


class NumpyDataNode(DataNode):
    def __init__(self, data):
        super(NumpyDataNode, self).__init__(data)

    @classmethod
    def from_numpy(cls, numpy_array):
        return cls(numpy_array)

    def _add_data_according2type(self):
        if self._meta_data_model.get_meta_data('dtype') is ndarray:
            self._data: ndarray
            self._meta_data_model.add_meta_data('dimension', self._data.shape)
            self._meta_data_model.add_meta_data('size', self._data.size)

            return
        raise TypeError('dtype is not recognized')

    def _check_input_data_type(self):
        if self._meta_data_model.get_meta_data('dtype') != ndarray:
            raise TypeError('not a numpy array type')

        return True


class PandasDataFrameDataNode(DataNode):
    def __init__(self, data):
        super(PandasDataFrameDataNode, self).__init__(data)

    @classmethod
    def from_pandas(cls, dataframe):
        return cls(dataframe)

    def _add_data_according2type(self):
        if self._meta_data_model.get_meta_data('dtype') is DataFrame:
            self._data: DataFrame
            self._meta_data_model.add_meta_data('dimension', self._data.shape)
            self._meta_data_model.add_meta_data('size', self._data.size)

            return
        raise TypeError('dtype is not recognized')

    def _check_input_data_type(self):
        if self._meta_data_model.get_meta_data('dtype') != DataFrame:
            raise TypeError('not a pandas dataframe type')


class PandasSeriesDataNode(DataNode):
    def __init__(self, data):
        super(PandasSeriesDataNode, self).__init__(data)

    @classmethod
    def from_pandas(cls, series):
        return cls(series)

    def _add_data_according2type(self):
        if self._meta_data_model.get_meta_data('dtype') is pd.Series:
            self._data: pd.Series
            self._meta_data_model.add_meta_data('dimension', self._data.shape)
            self._meta_data_model.add_meta_data('size', self._data.size)

            return
        raise TypeError('dtype is not recognized')

    def _check_input_data_type(self):
        if self._meta_data_model.get_meta_data('dtype') != pd.Series:
            raise TypeError('not a pandas series type')


class Task:
    def __init__(self, func: Callable):
        """
        basic unit in data pipeline
        first argument must be named as "data" or contains "data" in arguments

        Parameters
        ----------
        func
        """
        self._task_func = self.register_func(func)

    @property
    def name(self):
        from functools import partial
        if type(self._task_func) == partial:
            self._task_func: partial
            return self._task_func.func.__name__
        return self._task_func.__name__

    @property
    def description(self):
        return

    def register_func(self, func):
        import inspect
        if 'data' not in inspect.signature(func).parameters.keys():
            raise KeyError('"data" must be one of the input arguments in the function')
        return func

    def register_args(self, *args, **kwargs):
        from functools import partial
        self._task_func = partial(self._task_func, *args, **kwargs)
        return

    def execute(self, data_node: DataNode):
        return self._task_func(data=data_node.data)


class Step:
    def __init__(self, name: str, tasks: List = None, description: str = ""):
        self._name = name
        self._tasks = self._convert_tasks_before_creation(tasks)
        self._description = description

    @staticmethod
    def _convert_tasks_before_creation(tasks):
        if tasks is None:
            tasks = []
            return tasks
        else:
            raise_if_incorrect_type(tasks, Iterable)
            raise_if_not_all_type_uniform(tasks, Task)
            return tasks

    @property
    def name(self):
        return self._name

    @property
    def tasks(self):
        return self._tasks

    def tasks_count(self):
        return len(self.tasks)

    def get_tasks_name(self):
        return list(map(lambda x: x.name, self.tasks))

    def register_task(self, task):
        self._tasks.append(task)
        return


class Pipeline:
    """
    components in pipeline consist of multiple steps
    each step consists of multiple task

    more intuitive illustration
    Pipeline = [StepsA[TaskAA, TaskAB, TaskAC],
                StepsB[TaskBA, TaskBB],
                StepsC[TaskCA]]

    you are required to pass the DataNodeInferringEngine unless your data is using [pd.DataFrame, np.ndarray, pd.Series]
    DataNodeInferringEngine allow you to do customization of mapping in terms of data type,
    it will empower Pipeline to handle the data output in each tasks, steps
    then meta data can be created in each task, steps
    """

    def __init__(self, data_node_inferring_engine=None):
        self._pipeline_steps = SliceableDeque()
        self._pipeline_snapshot_res = []
        self._data_node_inferring_engine = data_node_inferring_engine

    @property
    def steps_name(self):
        return list(map(lambda x: x.name, self._pipeline_steps))

    @property
    def tasks_name(self):
        return list(map(lambda x: x.get_tasks_name(), self._pipeline_steps))

    def get_step_task_mapping(self):
        return dict(enumerate(zip(self.steps_name, self.tasks_name)))

    def get_pipeline_snapshot_res(self, step_id=None, task_id=None):
        if step_id is None:
            return self._pipeline_snapshot_res
        else:
            if task_id is None:
                return self._pipeline_snapshot_res[step_id]
            else:
                return self._pipeline_snapshot_res[step_id][task_id].data

    def reset_pipeline(self):
        self._pipeline_steps = SliceableDeque()
        return

    def remove_step(self, location_id: int):
        del self._pipeline_steps[location_id]
        return

    def insert_step(self, location_id: int, step: Step):
        self._pipeline_steps.insert(location_id, step)
        return

    def register_step(self, step: Step):
        self._pipeline_steps.append(step)

    def get_step(self, step_id):
        return self._pipeline_steps[step_id]

    def _clear_pipeline_snapshot_res(self):
        self._pipeline_snapshot_res = []
        return

    def execute(self, data_node: DataNode):
        self._clear_pipeline_snapshot_res()
        from copy import deepcopy
        res = deepcopy(data_node)
        for i, step in enumerate(self._pipeline_steps):
            tmp_res = []
            for ii, task in enumerate(step.tasks):
                res = task.execute(res)
                res = infer_DataNode_type(res, self._data_node_inferring_engine)
                tmp_res.append(res)
            self._pipeline_snapshot_res.append(tmp_res)
        return res
