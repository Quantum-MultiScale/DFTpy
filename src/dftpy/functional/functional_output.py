# Class handling output of functional evaluations
# general python imports
import copy

from dftpy.field import DirectField


class FunctionalOutput(object):
    """
    Object handling DFT functional output

    Attributes
    ----------
    name: string
        The (optional) name of the functional

    energy: float
        The energy

    potential: DirectField
        The first functional derivative of the functional wrt
        the electron density

    kernel: ReciprocalField
        The value of the reciprocal space kernel. This will
        be populated only if the functional is nonlocal
    """

    def __init__(
        self,
        name=None,
        energy=None,
        potential=None,
        energydensity=None,
        v2rho2=None,
        v3rho3=None,
        v4rho4=None,
        force=None,
        stress=None,
    ):

        args = locals()

        for key, value in args.items():
            if value is not None:
                setattr(self, key, value)
            elif key == "name":
                raise AttributeError("Functional name must be specified")
        self.attr_list = [
            'energy',
            'potential',
            'energydensity',
            'v2rho2',
            'v3rho3',
            'v4rho4',
            'force',
            'stress',
        ]

    def __iter__(self):
        for key in self.attr_list:
            if hasattr(self, key):
                yield key, getattr(self, key)

    def sum(self, other=None):
        if other is None:
            return self.copy()
        if self.name == other.name:
            name = self.name
        else:
            name = self.name + other.name
        result = FunctionalOutput(name=name)
        for key, value in self:
            if hasattr(other, key):
                setattr(result, key, value + getattr(other, key))
            else:
                setattr(result, key, value)
        for key, value in other:
            if not hasattr(result, key):
                setattr(result, key, value)
        return result

    def sub(self, other):
        result = self.copy()
        for key, value in self:
            if hasattr(other, key):
                setattr(result, key, value - getattr(other, key))
            else:
                setattr(result, key, value)
        for key, value in other:
            if not hasattr(result, key):
                setattr(result, key, -value)
        return result

    def mul(self, x):
        result = self.copy()
        for key, value in result:
            setattr(result, key, value * x)
        return result

    def div(self, x):
        if x == 0:
            raise ValueError("Dividing zero")
        result = self.copy()
        for key, value in result:
            setattr(result, key, value / x)
        return result

    def __add__(self, other):
        return self.sum(other)

    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, x):
        return self.mul(x)

    def __truediv__(self, x):
        return self.div(x)

    def copy(self):
        return copy.deepcopy(self)


class ZeroFunctional(object):
    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return 'ZeroFunctional'

    def __call__(self, rho, *args, **kwargs):
        return self.compute(rho, *args, **kwargs)

    @staticmethod
    def compute(rho, *args, calcType={'E', 'V'}, **kwargs):
        out = FunctionalOutput(name="ZERO", energy=0.0)
        if 'D' in calcType:
            out.energydensity = DirectField(grid=rho.grid)
        if 'V' in calcType:
            out.potential = DirectField(grid=rho.grid)
        return out
