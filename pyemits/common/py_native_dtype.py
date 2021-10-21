"""
this module stores the customized python native data type for performance
"""
import itertools
from collections import deque
import sys


class SliceableDeque(deque):
    """
    allow deque to do slicing, like the operation of list

    References
    ----------
    https://stackoverflow.com/questions/7064289/use-slice-notation-with-collections-deque/40315376

    Examples
    --------
    >>> d = SliceableDeque()
    >>> d.extend([1,2,3,4,5])
    >>> d[1:2]
    ... 2
    >>> d[slice(1, 2, None)]
    ... 2

    """

    def __getitem__(self, s):
        """

        Parameters
        ----------
        s: slice
            slice function, like slice(1,100,1), equivalent to 1:100:1

        Returns
        -------

        """
        try:
            start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
        except AttributeError:  # not a slice but an int
            return super().__getitem__(s)
        try:  # normal slicing
            return list(itertools.islice(self, start, stop, step))
        except ValueError:  # in case of a negative slice object
            length = len(self)
            start, stop = length + start if start < 0 else start, length + stop if stop < 0 else stop
            return list(itertools.islice(self, start, stop, step))
