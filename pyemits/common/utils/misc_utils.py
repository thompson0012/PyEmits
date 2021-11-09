from typing import Callable, Dict, Iterable, Union


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


def slice_iterables(func: Callable, *iterables: Iterable, return_list=False):
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


def get_class_attributes(cls_obj, ignore_startswith='default'):
    if ignore_startswith == 'default':
        ignore_startswith = ['__', '_']

    result = dir(cls_obj)

    if ignore_startswith is None:
        return result

    for i in ignore_startswith:
        result = list(filter(lambda x: not x.startswith(i), result))

    return result


def parallel_it(func: Callable,
                func_args: Union[Iterable, zip],
                element_type: str = 'auto_infer'):
    """
    Examples
    --------
    >>> import numpy as np
    >>> def create_fake_np(low, size):
    ...     return np.random.randint(low, size=size)

    >>> res = parallel_it(create_fake_np, zip([10000]*100,[100000]*100))
    >>> res = parallel_it(create_fake_np, list(zip([10000]*100000,[100000]*100000)))
    >>> res = parallel_it(create_fake_np, [dict(low=10000,size=100000)]*100)

    Parameters
    ----------
    func:

    func_args:

    element_type: str

    Returns
    -------

    """
    from joblib import Parallel, delayed
    from pyemits.common.validation import raise_if_incorrect_type
    from typing import Iterable
    raise_if_incorrect_type(func_args, Iterable)

    if element_type == 'auto_infer':
        if type(func_args) == 'zip':
            # zip element must be tuple like
            res = Parallel(n_jobs=-1)(delayed(func)(*args) for args in func_args)
            return res

        elif isinstance(func_args, (tuple, list)):
            # dict type
            # keywords args
            if isinstance(func_args[0], dict):
                res = Parallel(n_jobs=-1)(delayed(func)(**kwargs) for kwargs in func_args)
                return res

            # others, apart from dict type
            # positional args
            res = Parallel(n_jobs=-1)(delayed(func)(*args) for args in func_args)
            return res

    if element_type == 'tuple':
        res = Parallel(n_jobs=-1)(delayed(func)(*args) for args in func_args)
        return res
    elif element_type == 'dict':
        res = Parallel(n_jobs=-1)(delayed(func)(**kwargs) for kwargs in func_args)
        return res

    raise KeyError(f'element type: {element_type} is not acceptable')
