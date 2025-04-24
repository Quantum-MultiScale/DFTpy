import builtins
import sys

import numpy as np

from .mpi import MP, PMI, MPIFile, SerialComm
from .utils import mp, pmi, sprint

builtins.__dict__['sprint'] = sprint

# numpy array print without truncation
np.set_printoptions(threshold=sys.maxsize)
