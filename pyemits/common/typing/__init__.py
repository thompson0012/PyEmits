from enum import Enum


class StrEnum(str, Enum):
    @classmethod
    def to_list(cls) -> list:
        return list(map(lambda i: i.value, cls))
