"""Utility decorator for sharing docstrings across functions."""

from typing import Any, Callable


def doc(source: Any) -> Callable[[Callable], Callable]:
    """Return decorator that copies ``__doc__`` from *source*.

    Parameters
    ----------
    source : Any
        Object providing the docstring. If a string is supplied it is used
        directly. Otherwise the ``__doc__`` attribute is read from the
        object.
    """

    docstring = source if isinstance(source, str) else getattr(source, "__doc__", None)

    def decorator(func: Callable) -> Callable:
        func.__doc__ = docstring
        return func

    return decorator
