import functools
import sys
from contextlib import ExitStack

from dftpy.utils.utils import *


def lazymethod(meth):
    """Decorator for lazy evaluation and caching of data.

    Example::

      class MyClass:

         @lazymethod
         def thing(self):
             return expensive_calculation()

    The method body is only executed first time thing() is called, and
    its return value is stored.  Subsequent calls return the cached
    value."""
    name = meth.__name__

    @functools.wraps(meth)
    def getter(self):
        try:
            cache = self._lazy_cache
        except AttributeError:
            cache = self._lazy_cache = {}

        if name not in cache:
            cache[name] = meth(self)
        return cache[name]

    return getter


def lazyproperty(meth):
    """Decorator like lazymethod, but making item available as a property."""
    return property(lazymethod(meth))


class IOContext(object):
    @lazyproperty
    def _exitstack(self):
        return ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def closelater(self, fd):
        return self._exitstack.enter_context(fd)

    def close(self):
        self._exitstack.close()

    def openfile(self, file, comm=None, mode='w'):
        from dftpy.mpi import mp

        if comm is None:
            comm = mp.comm

        if hasattr(file, 'close'):
            return file  # File already opened, not for us to close.

        if file is None or comm.rank != 0:
            return self.closelater(open(os.devnull, mode=mode))

        if file == '-':
            return sys.stdout

        return self.closelater(open(file, mode=mode))
