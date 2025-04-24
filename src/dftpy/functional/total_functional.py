import numpy as np

from dftpy.functional.abstract_functional import AbstractFunctional
from dftpy.functional.functional_output import FunctionalOutput


class TotalFunctional(AbstractFunctional):
    """
    Object handling energy evaluation for the
    purposes of optimizing the electron density

    Attributes
    ----------

    KineticEnergyFunctional, XCFunctional, PSEUDO, HARTREE: Functional
        Instances of functional class needed for the computation
        of the chemical potential, total potential and total energy.

    Example
    -------

    KE = Functional(type='KEDF',name='TF')
    XC = Functional(type='XC',name='LDA')
    PSEUDO = Functional(type='PSEUDO', kwargs)
    HARTREE = Functional(type='HARTREE')

    EnergyEvaluator = TotalEnergyAndPotential(
        KineticEnergyFunctional = KE,
        XCFunctional = XC,
        PSEUDO = PSEUDO,
        HARTREE = HARTREE
    )

    or given a dict
    funcdict = {
        "KineticEnergyFunctional": KE,
        "XCFunctional": XC,
        "PSEUDO": PSEUDO,
        "HARTREE": HARTREE
    }
    EnergyEvaluator = TotalEnergyAndPotential(**funcdict)

    [the energy:]
    E = EnergyEvaluator.Energy(rho)

    [total energy and potential:]
    out = EnergyEvaluator.compute(rho)

    """

    def __init__(self, ewald=None, **kwargs):

        self.funcDict = {}
        self.funcDict.update(kwargs)
        # remove useless key
        for key, evalfunctional in list(self.funcDict.items()):
            if evalfunctional is None:
                del self.funcDict[key]

        self.UpdateNameType()
        self._ewald = ewald

    def __repr__(self):
        return self.funcDict.__repr__()

    def __getitem__(self, key):
        return self.funcDict[key]

    def __setitem__(self, key, value):
        self.funcDict.update({key: value})

    def UpdateNameType(self):
        self.name = ""
        self.type = ""
        for key, evalfunctional in self.funcDict.items():
            if not isinstance(evalfunctional, AbstractFunctional):
                raise TypeError("{} must be AbstractFunctional".format(key))
            setattr(self, key, evalfunctional)
            self.name += getattr(evalfunctional, 'name') + " "
            self.type += getattr(evalfunctional, 'type') + " "

    def UpdateFunctional(self, keysToRemove=[], newFuncDict={}):
        for key in keysToRemove:
            self.funcDict.pop(key, None)

        self.funcDict.update(newFuncDict)
        self.UpdateNameType()

    def Subset(self, keys):
        subdict = dict((key, self.funcDict[key]) for key in keys)
        return TotalFunctional(**subdict)

    @property
    def ewald(self):
        """
        Don't save the ewald because it maybe update in PSEUDO.
        """
        if self._ewald is None:
            for key, func in self.funcDict.items():
                if func.type == "PSEUDO":
                    ewald = func.get_ewald()
                    break
            else:
                ewald = None
        else:
            ewald = self._ewald
        return ewald

    def compute(self, rho, calcType={"E", "V"}, **kwargs):
        Obj = self.get_energy_potential(rho, calcType=calcType, **kwargs)
        return Obj

    def Energy(self, rho, **kwargs):
        energy = self.compute(rho, calcType={"E"}).energy
        return energy

    def get_energy_potential(self, rho, calcType={"E", "V"}, split=False, **kwargs):
        energy_potential = {}
        energy_potential["TOTAL"] = FunctionalOutput(name="TOTAL", energy=0.0)
        for key, func in self.funcDict.items():
            if split:
                if func.type == "KEDF":
                    results = func(rho, calcType=calcType, split=split)
                    for key2 in results:
                        k = "KEDF-" + key2.split('-')[-1]
                        energy_potential[k] = results[key2] + energy_potential.get(
                            k, None
                        )
                        energy_potential["TOTAL"] += results[key2]
                else:
                    results = func(rho, calcType=calcType, **kwargs)
                    energy_potential[func.type] = results
                    energy_potential["TOTAL"] += results
            else:
                results = func(rho, calcType=calcType, **kwargs)
                energy_potential["TOTAL"] += results
        #
        if self.ewald:
            results = FunctionalOutput(name="II", energy=self.ewald.energy)
            if split:
                energy_potential["II"] = results
            energy_potential["TOTAL"] += results
        #
        if 'E' in calcType:
            keys, ep = zip(*energy_potential.items())
            values = [item.energy for item in ep]
            values = rho.mp.vsum(values)
            for key, v in zip(keys, values):
                energy_potential[key].energy = v
        if not split:
            energy_potential = energy_potential["TOTAL"]
        return energy_potential

    def get_forces(self, rho, ions=None, split=False, **kwargs):
        if ions is None:
            if self.ewald:
                ions = self.ewald.ions
            else:
                raise ValueError('Please provide ions for forces')
        forces = {}
        forces["TOTAL"] = np.zeros((ions.nat, 3))
        for key, func in self.funcDict.items():
            if hasattr(func, 'forces'):
                f = func.forces(rho)
                if f is not None:
                    if split:
                        forces[key] = f
                    forces["TOTAL"] += f
        #
        if self.ewald:
            f = self.ewald.forces
            if split:
                forces['II'] = f
            forces["TOTAL"] += f
        #
        if split:
            for key, f in forces.items():
                forces[key] = rho.mp.vsum(f)
        else:
            forces["TOTAL"] = rho.mp.vsum(forces["TOTAL"])
        if not split:
            forces = forces["TOTAL"]
        return forces

    def get_stress(self, rho, split=False, **kwargs):
        """
        Get stress tensor
        """
        # -----------------------------------------------------------------------
        stress = {}
        stress['TOTAL'] = np.zeros((3, 3))

        for key, func in self.funcDict.items():
            if func.type == "KEDF":
                results = func.stress(rho, split=True)
                for key2 in results:
                    stress["TOTAL"] += results[key2]
                    k = "KEDF-" + key2.split('-')[-1]
                    stress[k] = stress.get(k, 0.0) + results[key2]
            else:
                results = func.stress(rho)
                stress["TOTAL"] += results
                stress[func.type] = results

        if self.ewald:
            stress['II'] = self.ewald.stress
            stress["TOTAL"] += stress['II']

        for i in range(1, 3):
            for j in range(i - 1, -1, -1):
                stress["TOTAL"][i, j] = stress["TOTAL"][j, i]

        keys, values = zip(*stress.items())
        values = rho.mp.vsum(values)
        stress = dict(zip(keys, values))

        if not split:
            stress = stress["TOTAL"]
        return stress
