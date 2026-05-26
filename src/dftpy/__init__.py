from .config import *
from .mpi import mp, sprint
from .time_data import TimeData, timer

try:
    from importlib.metadata import version # python >= 3.8
    __version__ = version("dftpy")
except Exception:
    try:
        from .version import __version__
    except Exception:
        __version__ = '0.0.0'
