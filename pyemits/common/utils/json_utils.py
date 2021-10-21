import json
from abc import ABC, abstractmethod
from typing import Union, List


class JsonFinderBase(ABC):
    def __init__(self, json_obj):
        self.json_obj = json_obj

    @classmethod
    def from_json_str(cls, json_str):
        """
        factory method for creating JasonPathFinder
        Returns
        -------

        """
        json_obj = json.loads(json_str)
        return cls(json_obj)

    @classmethod
    def from_json_dict(cls, json_dict):
        json_obj = json_dict
        return cls(json_obj)

    @abstractmethod
    def iter_node(self, rows: Union[dict, list], road_step, target: Union[str, int]):
        pass

    def find_one(self, key: str) -> list:
        path_iter = self.iter_node(self.json_obj, [], key)
        for path in path_iter:
            return path
        return []

    def find_all(self, key: str) -> List[list]:
        path_iter = self.iter_node(self.json_obj, [], key)
        return list(path_iter)


class JsonPathFinder(JsonFinderBase):
    def __init__(self, json_obj):
        super(JsonPathFinder, self).__init__(json_obj)

    def iter_node(self, rows: Union[dict, list], road_step, target: Union[str, int]):
        if isinstance(rows, dict):
            key_value_iter = (x for x in rows.items())
        elif isinstance(rows, (list, tuple)):
            key_value_iter = (x for x in enumerate(rows))
        else:
            return
        for key, value in key_value_iter:
            current_path = road_step.copy()
            current_path.append(key)
            if key == target:
                yield current_path
            if isinstance(value, (dict, list, tuple)):
                yield from self.iter_node(value, current_path, target)


class JsonValueFinder(JsonFinderBase):
    def __init__(self, json_obj):
        super(JsonValueFinder, self).__init__(json_obj)

    def iter_node(self, rows: Union[dict, list], road_step, target: Union[str, int]):
        if isinstance(rows, dict):
            key_value_iter = (x for x in rows.items())
        elif isinstance(rows, (list, tuple)):
            key_value_iter = (x for x in enumerate(rows))
        else:
            return
        for key, value in key_value_iter:
            # current_path = road_step.copy() # not useful for value finder
            # current_path.append(key) # not useful for value finder
            if key == target:
                yield value
            if isinstance(value, (dict, list, tuple)):
                yield from self.iter_node(value, [], target)
