"""
make dictionary accessible like javascript
alternative solution: Munch, attrdict

Examples:
>>> obj = {'foo': [{'a': 1,'b': {'c': 2,}],'bar': 5}
>>> attrobj = wrap(obj)
>>> assert attrobj.foo.a == 1
"""


# https://stackoverflow.com/a/31569634/1210352
class DictProxy(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return wrap(self.obj[key])

    def __getattr__(self, key):
        try:
            return wrap(getattr(self.obj, key))
        except AttributeError:
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

    # you probably also want to proxy important list properties along like
    # items(), iteritems() and __len__


class ListProxy(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        return wrap(self.obj[key])

    # you probably also want to proxy important list properties along like
    # __iter__ and __len__


def wrap(value):
    if isinstance(value, dict):
        return DictProxy(value)
    if isinstance(value, (tuple, list)):
        return ListProxy(value)
    return value
