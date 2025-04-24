import numpy as np

# from dftpy.field import DirectField, ReciprocalField


def population_analysis(rho, total_rho, method='hirshfeld', **kwargs):
    ratio = np.where(total_rho > 0.0, rho / total_rho, 0.0)
    return ratio
