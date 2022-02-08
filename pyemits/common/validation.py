"""
function gate checker for ensure input is correct
"""
from pyemits.common.errors import ItemNotFoundError
from typing import Union, Any, Sequence


def raise_if_incorrect_type(obj: object, expected_type: Any):
    """
    general function to check obj type
    raise error when it is incorrect

    Parameters
    ----------
    obj: Object
        any object

    expected_type: Any
        any class or type

    Returns
    -------
    true_false: bool
        only return True when matched type

    Raises
    ------
    TypeError

    """
    if isinstance(obj, expected_type):
        return True

    raise TypeError(f"obj {str(obj)} is not an expected type {str(expected_type)}")


def raise_if_value_not_contains(sequences, references):
    mapped_ = map(lambda x: x in references, sequences)
    if any(mapped_):
        return True

    raise ItemNotFoundError("item not found:")


def raise_if_not_all_value_contains(sequences, references):
    mapped = map(lambda x: x in references, sequences)
    if not all(mapped):
        raise ItemNotFoundError("item not contains:", list(mapped))

    return True


def raise_if_not_dataclass(obj):
    from dataclasses import is_dataclass

    if not is_dataclass(obj):
        raise TypeError(f"obj {str(obj)} is not a dataclass type")

    return True


def raise_if_not_all_element_type_uniform(iterable_sequences: Union[tuple, list], expect_type=None):
    if not check_all_element_type_uniform(iterable_sequences, expect_type):
        raise TypeError('elements type not the same')
    return True


def check_all_element_type_uniform(iterable_sequences: Union[tuple, list], expect_type=None):
    if not expect_type:
        first_type = type(iterable_sequences[0])
        expect_type = first_type
    return all(isinstance(item, expect_type) for item in iterable_sequences)


def check_if_value_between(value, expectation):
    if min(expectation) <= value <= max(expectation):
        return True

    return False


def raise_if_value_not_between(value, expectation):
    if check_if_value_between(value, expectation):
        return True
    raise ValueError(f'value not in expectation range, min: {min(expectation)}, max: {max(expectation)}')


def check_if_any_value_true(*args):
    if any([*args]):
        return True

    return False


def check_if_all_value_true(*args):
    if all([*args]):
        return True

    return False


def is_initialized_cls(obj) -> bool:
    import inspect
    if inspect.isclass(obj):
        return False  # still a class

    return True  # initialized


def raise_if_not_init_cls(obj):
    if is_initialized_cls(obj):
        return True

    raise TypeError('model must be initialized before any use')


def raise_if_values_not_same(iterables: Sequence):
    first_item = iterables[0]
    if not all(map(lambda x: x == first_item, iterables)):
        raise ValueError('sequences not same values')

    return True
