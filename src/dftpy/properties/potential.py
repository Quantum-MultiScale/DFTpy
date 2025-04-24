import numpy as np


def get_electrostatic_potential(rho, evaluator):
    vloc = evaluator.funcDict['PSEUDO']
    vhart = evaluator.funcDict['HARTREE']
    v = vloc(rho, calcType=['V']).potential + vhart(rho, calcType=['V']).potential
    return v
