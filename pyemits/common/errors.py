class BaseError(Exception):
    """
    root class of error
    """


class ItemNotFoundError(BaseError):
    """
    item not found error
    """


class ItemOverwriteError(BaseError):
    """
    item already exist,
    or not allow for overwrite
    """
