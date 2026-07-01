import numpy as np

from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput


class ExternalPotential(AbstractFunctional):

    def __init__(self, v=None):
        # if v is None:
        #    raise AttributeError("Must specify v")
        # else:
        self.name = 'EXT'
        self.type = 'EXT'
        self._v = v

    def __repr__(self):
        return 'EXT'

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        self._v = new_v

    def compute(self, rho, calcType={"E", "V"}, **kwargs):
        pot = self.v
        if 'E' in calcType:
            ene = (self.v * rho).integral()
        else:
            ene = 0
        return FunctionalOutput(name="ext", energy=ene, potential=pot)

    def potential(self):
        return self.v
