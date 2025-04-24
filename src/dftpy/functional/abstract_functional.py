# Class handling functional evaluations
# functional class (output handler) in output

# local imports
# from dftpy.mpi import sprint

# general python imports
from abc import ABC, abstractmethod

from dftpy.functional.functional_output import FunctionalOutput


class AbstractFunctional(ABC):
    def __repr__(self):
        name = getattr(self, 'name', self.__class__.__name__)
        rep = name + ', ' + self.__dict__.__repr__()
        return rep

    def __call__(self, rho, *args, **kwargs):
        return self.compute(rho, *args, **kwargs)

    @abstractmethod
    def compute(self, rho, *args, **kwargs):
        # returns energy and potential
        pass

    @property
    def name(self):
        if getattr(self, '_name', None) is None:
            self._name = self.type
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def type(self):
        if getattr(self, '_type', None) is None:
            self._type = self.__class__.__name__
        return self._type

    @type.setter
    def type(self, value):
        self._type = value
