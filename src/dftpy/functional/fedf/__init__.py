# Collection of finite temperature free energy density functional

# some funtion used in free energy density functional

import numpy as np

from dftpy.math_utils import PowerInt
from dftpy.time_data import timer
from dftpy.field import DirectField

__all__ = ['get_reduce_t','FTK','FTK_dt','FTK_dt2','FTK_dt3','FTH','FTH_dt','FTH_dt2',
           'FTZETA','FTZETA_dt','FTXI','FTXI_dt']

# reduced temperature
def get_reduce_t(rho,FT_T):
    t = 2 * FT_T / PowerInt(3 * np.pi**2 * rho,2,3)
    return t 

##parameters 

FTK_U = np.array([-2.5, -2.141088549, 0.2210798602, 0.0007916274395, -0.004351943569,
                  0.004188256879, -0.002144912720, 0.0005590314373, -0.00005824689694, 0.0])

FTK_D = np.array([1.0, -4.112335167, 1.995732255, 14.83844536, -178.4789624, 992.5850212,
                  -3126.965212, 5296.225924, -3742.224547, 0.0])

FTT0 = 0.543010717965

FTH_U = np.array([3.0, -0.7996705242, 0.2604164189, -0.1108908431, 0.06875811936,
                  -0.03515486636, 0.01002514804, -0.001153263119, 0.0, 0.0])

FTH_D = np.array([1.0, 3.210141829, 58.30028308, -887.5691412, 6055.757436,
                  -22429.59828, 43277.02562, -34029.06962, 0.0, 0.0])

### FTK and FTH 

def FTK(t):
    # Initialize K as an array of the same shape as t
    K = np.zeros_like(t)

    # Apply the formula element-wise depending on the value of t
    mask = t >= FTT0

    # For t >= FTT0, apply the first formula
    K[mask] = (FTK_U[0] * t[mask] * np.log(t[mask]) +
               FTK_U[1] * t[mask] +
               FTK_U[2] * t[mask]**(-0.5) +
               FTK_U[3] * t[mask]**(-2.0) +
               FTK_U[4] * t[mask]**(-3.5) +
               FTK_U[5] * t[mask]**(-5.0) +
               FTK_U[6] * t[mask]**(-6.5) +
               FTK_U[7] * t[mask]**(-8.0) +
               FTK_U[8] * t[mask]**(-9.5))

    # For t < FTT0, apply the second formula
    K[~mask] = (FTK_D[0] +
                FTK_D[1] * t[~mask]**(2.0) +
                FTK_D[2] * t[~mask]**(4.0) +
                FTK_D[3] * t[~mask]**(6.0) +
                FTK_D[4] * t[~mask]**(8.0) +
                FTK_D[5] * t[~mask]**(10.0) +
                FTK_D[6] * t[~mask]**(12.0) +
                FTK_D[7] * t[~mask]**(14.0) +
                FTK_D[8] * t[~mask]**(16.0))

    return K

def FTK_dt(t):
    K = np.zeros_like(t)

    mask = t >= FTT0

    K[mask] = (FTK_U[0] * (np.log(t[mask]) + 1) +
               FTK_U[1] * 1 +
               -0.5 * FTK_U[2] * t[mask]**(-1.5) +
               -2.0 * FTK_U[3] * t[mask]**(-3.0) +
               -3.5 * FTK_U[4] * t[mask]**(-4.5) +
               -5.0 * FTK_U[5] * t[mask]**(-6.0) +
               -6.5 * FTK_U[6] * t[mask]**(-7.5) +
               -8.0 * FTK_U[7] * t[mask]**(-9.0) +
               -9.5 * FTK_U[8] * t[mask]**(-10.5))

    K[~mask] = (0 +
                2.0 * FTK_D[1] * t[~mask]**(1.0) +
                4.0 * FTK_D[2] * t[~mask]**(3.0) +
                6.0 * FTK_D[3] * t[~mask]**(5.0) +
                8.0 * FTK_D[4] * t[~mask]**(7.0) +
                10.0 * FTK_D[5] * t[~mask]**(9.0) +
                12.0 * FTK_D[6] * t[~mask]**(11.0) +
                14.0 * FTK_D[7] * t[~mask]**(13.0) +
                16.0 * FTK_D[8] * t[~mask]**(15.0))

    return K

def FTK_dt2(t):
    K = np.zeros_like(t)
    mask = t >= FTT0
    K[mask] = (
            FTK_U[0] * (1.0 / t)
            + 0.0
            + (-0.5 * -1.5)   * FTK_U[2] * t[mask]**(-2.5)
            + (-2.0 * -3.0)   * FTK_U[3] * t[mask]**(-4.0)
            + (-3.5 * -4.5)   * FTK_U[4] * t[mask]**(-5.5)
            + (-5.0 * -6.0)   * FTK_U[5] * t[mask]**(-7.0)
            + (-6.5 * -7.5)   * FTK_U[6] * t[mask]**(-8.5)
            + (-8.0 * -9.0)   * FTK_U[7] * t[mask]**(-10.0)
            + (-9.5 * -10.5)  * FTK_U[8] * t[mask]**(-11.5)
        )

    K[~mask]= ( 0.0
            + 2.0  * FTK_D[1]
            + 4.0  * 3.0  * FTK_D[2] * t[~mask]**(2.0)
            + 6.0  * 5.0  * FTK_D[3] * t[~mask]**(4.0)
            + 8.0  * 7.0  * FTK_D[4] * t[~mask]**(6.0)
            + 10.0 * 9.0  * FTK_D[5] * t[~mask]**(8.0)
            + 12.0 * 11.0 * FTK_D[6] * t[~mask]**(10.0)
            + 14.0 * 13.0 * FTK_D[7] * t[~mask]**(12.0)
            + 16.0 * 15.0 * FTK_D[8] * t[~mask]**(14.0)
        )

    return K

def FTK_dt3(t):
    K = np.zeros_like(t)
    mask = t >= FTT0
    K[mask] = (
            -FTK_U[0] * (1.0 / t[mask]**2)
            + 0.0
            + (-0.5 * -1.5 * -2.5)    * FTK_U[2] * t[mask]*(-3.5)
            + (-2.0 * -3.0 * -4.0)    * FTK_U[3] * t[mask]*(-5.0)
            + (-3.5 * -4.5 * -5.5)    * FTK_U[4] * t[mask]*(-6.5)
            + (-5.0 * -6.0 * -7.0)    * FTK_U[5] * t[mask]*(-8.0)
            + (-6.5 * -7.5 * -8.5)    * FTK_U[6] * t[mask]*(-9.5)
            + (-8.0 * -9.0 * -10.0)   * FTK_U[7] * t[mask]*(-11.0)
            + (-9.5 * -10.5 * -11.5)  * FTK_U[8] * t[mask]*(-12.5)
        )
    k[~mask]= (
            0.0
            + 0.0
            + 4.0  * 3.0  * 2.0   * FTK_D[2] * t[~mask]
            + 6.0  * 5.0  * 4.0   * FTK_D[3] * t[~mask]**3.0
            + 8.0  * 7.0  * 6.0   * FTK_D[4] * t[~mask]**5.0
            + 10.0 * 9.0  * 8.0   * FTK_D[5] * t[~mask]**7.0
            + 12.0 * 11.0 * 10.0  * FTK_D[6] * t[~mask]**9.0
            + 14.0 * 13.0 * 12.0  * FTK_D[7] * t[~mask]**11.0
            + 16.0 * 15.0 * 14.0  * FTK_D[8] * t[~mask]**13.0
        )
    return K

#### FTH 

def FTH(t):
    # Initialize H as an array of the same shape as t
    H = np.zeros_like(t)

    # Apply the formula element-wise depending on the value of t
    mask = t >= FTT0

    # For t >= FTT0, apply the first formula
    H[mask] = (FTH_U[0] +
               FTH_U[1] * t[mask]**(-1.5) +
               FTH_U[2] * t[mask]**(-3.0) +
               FTH_U[3] * t[mask]**(-4.5) +
               FTH_U[4] * t[mask]**(-6.0) +
               FTH_U[5] * t[mask]**(-7.5) +
               FTH_U[6] * t[mask]**(-9.0) +
               FTH_U[7] * t[mask]**(-10.5))

    # For t < FTT0, apply the second formula
    H[~mask] = (FTH_D[0] +
                FTH_D[1] * t[~mask]**(2.0) +
                FTH_D[2] * t[~mask]**(4.0) +
                FTH_D[3] * t[~mask]**(6.0) +
                FTH_D[4] * t[~mask]**(8.0) +
                FTH_D[5] * t[~mask]**(10.0) +
                FTH_D[6] * t[~mask]**(12.0) +
                FTH_D[7] * t[~mask]**(14.0))

    return H

def FTH_dt(t):
    # Initialize H as an array of the same shape as t
    H = np.zeros_like(t)

    # Create mask for t >= FTT0
    mask = t >= FTT0

    # For t >= FTT0
    H[mask] = (
        -1.5  * FTH_U[1] * t[mask]**(-2.5) +
        -3.0  * FTH_U[2] * t[mask]**(-4.0) +
        -4.5  * FTH_U[3] * t[mask]**(-5.5) +
        -6.0  * FTH_U[4] * t[mask]**(-7.0) +
        -7.5  * FTH_U[5] * t[mask]**(-8.5) +
        -9.0  * FTH_U[6] * t[mask]**(-10.0) +
        -10.5 * FTH_U[7] * t[mask]**(-11.5)
    )

    # For t < FTT0
    H[~mask] = (
        2.0  * FTH_D[1] * t[~mask]**(1.0) +
        4.0  * FTH_D[2] * t[~mask]**(3.0) +
        6.0  * FTH_D[3] * t[~mask]**(5.0) +
        8.0  * FTH_D[4] * t[~mask]**(7.0) +
        10.0 * FTH_D[5] * t[~mask]**(9.0) +
        12.0 * FTH_D[6] * t[~mask]**(11.0) +
        14.0 * FTH_D[7] * t[~mask]**(13.0)
    )

    return H

def FTH_dt2(t):
    # Initialize H as an array of the same shape as t
    H = np.zeros_like(t)

    # Create mask for t >= FTT0
    mask = t >= FTT0

    # For t >= FTT0
    H[mask] = (
        -1.5  * -2.5  * FTH_U[1] * t[mask]**(-3.5) +
        -3.0  * -4.0  * FTH_U[2] * t[mask]**(-5.0) +
        -4.5  * -5.5  * FTH_U[3] * t[mask]**(-6.5) +
        -6.0  * -7.0  * FTH_U[4] * t[mask]**(-8.0) +
        -7.5  * -8.5  * FTH_U[5] * t[mask]**(-9.5) +
        -9.0  * -10.0 * FTH_U[6] * t[mask]**(-11.0) +
        -10.5 * -11.5 * FTH_U[7] * t[mask]**(-12.5)
    )

    # For t < FTT0
    H[~mask] = (
        2.0  * 1.0  * FTH_D[1] +
        4.0  * 3.0  * FTH_D[2] * t[~mask]**(2.0) +
        6.0  * 5.0  * FTH_D[3] * t[~mask]**(4.0) +
        8.0  * 7.0  * FTH_D[4] * t[~mask]**(6.0) +
        10.0 * 9.0  * FTH_D[5] * t[~mask]**(8.0) +
        12.0 * 11.0 * FTH_D[6] * t[~mask]**(10.0) +
        14.0 * 13.0 * FTH_D[7] * t[~mask]**(12.0)
    )

    return H

### zeta and xi 

def FTZETA(t):
    park = FTK_dt(t)
    return -t*park 

def FTZETA_dt(t): 
    park = FTK_dt(t)
    ppark = FTK_dt2(t)
    return -t*ppark - park

def FTXI(t): 
    park= FTK_dt(t) 
    k = FTK(t)
    return  k-t*park 

def FTXI_dt(t):
    ppark = FTK_dt2(t) 
    return -t*ppark 
