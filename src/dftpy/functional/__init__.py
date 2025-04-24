from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.external_potential import ExternalPotential
from dftpy.functional.functional_output import FunctionalOutput
from dftpy.functional.hartree import Hartree
from dftpy.functional.kedf import KEDF
from dftpy.functional.nonadiabatic import Dynamic
from dftpy.functional.pseudo import LocalPseudo
from dftpy.functional.semilocal_xc import XC
from dftpy.functional.total_functional import TotalFunctional

FunctionalTypeDict = {
    'KEDF': KEDF,
    'PSEUDO': LocalPseudo,
    'HARTREE': Hartree,
    'XC': XC,
    'EXT': ExternalPotential,
    'DYNAMIC': Dynamic,
}


def Functional(type=None, optional_kwargs=None, **kwargs):
    """
    Function that instantiate objects which handle the evaluation of a DFT functional

    Attributes
    ----------
    name: string
        The name of the functional
    type: string
        The functional type (XC, KEDF, HARTREE, PSEUDO)
    optional_kwargs: dict
        set of kwargs for the different functional types/names

    Examples
    --------
     XC = Functional(type='XC',name='LDA')
     outXC = XC(rho)
     outXC.energy --> the energy
     outXC.potential     --> the pot
    """
    type = type.upper()
    if optional_kwargs is not None:
        kwargs.update(optional_kwargs)

    try:
        return FunctionalTypeDict[type](**kwargs)
    except KeyError:
        raise AttributeError("Unknown functional type: {type}.")
