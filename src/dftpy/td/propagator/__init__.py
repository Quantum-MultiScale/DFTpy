from typing import Union

from dftpy.td.propagator.crank_nicholson import CrankNicholson
from dftpy.td.propagator.taylor import Taylor


class Propagator(object):
    """
    Wrapper for other propagator classes, so you can create an object for any propagator class.
    Examples:
    Propagator(name='CrankNicholson', hamiltonian=hamiltonian, interval=0.1)
    is the same as
    CrankNicholson(hamiltonian=hamiltonian, interval=0.1)
    """

    NameDict = {
        'taylor': Taylor,
        'crank-nicholson': CrankNicholson,
    }

    def __new__(cls, *args, name: Union[str, None] = None, **kwargs):
        """
        Parameters
        ----------
        args:
            Arguments passed to the propagator
        name : str
            Name of the propagator, options:
            'taylor', 'crank-nicholson'
        **kwargs : dict
            Keyword arguments passed to the propagator
        """
        try:
            return cls.NameDict[name](*args, **kwargs)
        except KeyError:
            raise AttributeError(
                "{0:s} is not a supported propagator type".format(name)
            )
