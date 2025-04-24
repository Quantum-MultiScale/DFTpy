import copy
from typing import Optional, Set

from dftpy.field import DirectField
from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.nonadiabatic.jp import JP1
from dftpy.functional.nonadiabatic.wcdhc import WCDHC, nWCDHC
from dftpy.mpi.utils import sprint

__all__ = ["Dynamic"]

Dynamic_Dict = {
    "WCDHC": WCDHC,
    "nWCDHC": nWCDHC,
    "JP1": JP1,
}


class Dynamic(AbstractFunctional):
    def __init__(self, name: str, **kwargs) -> None:
        self.type = 'DYNAMIC'
        if name not in Dynamic_Dict:
            raise AttributeError(
                "{0:s} is not a correct dynamic functional name.".format(name)
            )
        self.name = name
        self.kwargs = kwargs

    def compute(
        self,
        density: DirectField,
        current: DirectField = None,
        der_current: Optional[DirectField] = None,
        calcType: Set[str] = ["V"],
        **kwargs,
    ) -> FunctionalOutput:
        kw_args = copy.deepcopy(self.kwargs)
        kw_args.update(kwargs)
        return Dynamic_Dict[self.name](
            density, current, der_current=der_current, calcType=calcType, **kw_args
        )
