from abc import ABC, abstractmethod

import numpy as np

from dftpy.constants import environ

if environ["FFTLIB"] == "pyfftw":
    """
    pyfftw.config.NUM_THREADS  =  multiprocessing.cpu_count()
    print('threads', pyfftw.config.NUM_THREADS)
    """
    import pyfftw


class AbstractFFT(ABC):
    """
    This is a pseudo potential template class and should never be touched.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def fft(self):
        pass


class PYfft(AbstractFFT):
    def __init__(self, grid, cplx=False, threads=1, flags=("FFTW_MEASURE",), **kwargs):
        self.shapes = [np.zeros(3), np.zeros(3)]
        self.obj = [None, None]
        self.threads = threads
        self.flags = flags
        self.direction = "FFTW_FORWARD"

    def fft(self, grid, cplx=False):
        nr = grid.nr
        if np.all(nr == self.shapes[cplx]):
            fft_object = self.obj[cplx]
        else:
            if cplx:
                rA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
                cA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
            else:
                nrc = grid.nrG
                rA = pyfftw.empty_aligned(tuple(nr), dtype="float64")
                cA = pyfftw.empty_aligned(tuple(nrc), dtype="complex128")
            fft_object = pyfftw.FFTW(
                rA,
                cA,
                axes=(0, 1, 2),
                flags=self.flags,
                direction=self.direction,
                threads=self.threads,
            )
            self.shapes[cplx] = nr
            self.obj[cplx] = fft_object
        return fft_object


class PYIfft(AbstractFFT):
    def __init__(self, nr=[0, 0, 0], cplx=False, threads=1, **kwargs):
        self.shapes = [np.zeros(3), np.zeros(3)]
        self.obj = [None, None]
        self.threads = threads
        self.flags = ("FFTW_MEASURE",)
        self.direction = "FFTW_BACKWARD"

    def fft(self, grid, cplx=False):
        nr = grid.nrR
        if np.all(nr == self.shapes[cplx]):
            fft_object = self.obj[cplx]
        else:
            if cplx:
                rA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
                cA = pyfftw.empty_aligned(tuple(nr), dtype="complex128")
            else:
                nrc = grid.nr
                rA = pyfftw.empty_aligned(tuple(nr), dtype="float64")
                cA = pyfftw.empty_aligned(tuple(nrc), dtype="complex128")
            fft_object = pyfftw.FFTW(
                cA,
                rA,
                axes=(0, 1, 2),
                flags=self.flags,
                direction=self.direction,
                threads=self.threads,
            )
            self.shapes[cplx] = nr
            self.obj[cplx] = fft_object
        return fft_object
