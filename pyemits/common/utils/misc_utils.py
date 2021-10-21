from typing import Callable, Dict, Iterable


def map_values(dictionary: dict, func: Callable):
    """

    Parameters
    ----------
    dictionary :
    func :

    Returns
    -------

    Examples
    -------
    >>> users = {'fred': { 'user': 'fred', 'age': 40 }, 'pebbles': { 'user': 'pebbles', 'age': 1 }}

    >>> map_values(users, lambda u : u['age'])
    {'fred': 40, 'pebbles': 1}

    >>> map_values(users, lambda u : u['age']+1)
    {'fred': 41, 'pebbles': 2}

    """
    ret = {}
    for key in dictionary.keys():
        ret[key] = func(dictionary[key])

    return ret


def slice_iterables(func: Callable, *iterables: Iterable , return_list=False):
    """
    Examples
    --------
    >>> def sum(a,b):
    ...    return a+b
    >>> lst1=[2,4,6,8]
    >>> lst2=[1,3,5,7,9]
    >>> result=list(map(sum,lst1,lst2))
    >>> print(result)
    ... [3, 7, 11, 15]

    Parameters
    ----------
    iterables_
    func

    Returns
    -------

    """
    if return_list:
        return list(map(func, *iterables))
    return map(func, *iterables)

