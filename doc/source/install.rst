.. _download_and_install:

============
Installation
============

Requirements
============

* Python_ >= 3.6
* NumPy_ >= 1.8.0
* SciPy_ >= 0.18.0
* ASE_  >= 3.22.0
* xmltodict_ >= 0.12.0

Optional:

* pylibxc_ (for using exchange-correlation functionals other than LDA)
* pyFFTW_  (for Fast Fourier Transform)
* upf_to_json_ (For UPF pseudopotential)
* mpi4py_ (MPI for python)
* mpi4py-fft_ (Fast Fourier Transforms with MPI)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _pylibxc: https://tddft.org/programs/libxc/
.. _pyFFTW: https://pyfftw.readthedocs.io/en/latest/
.. _ASE: https://gitlab.com/ase/ase
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py
.. _mpi4py-fft: https://bitbucket.org/mpi4py/mpi4py-fft
.. _xmltodict: https://github.com/martinblech/xmltodict
.. _upf_to_json: https://github.com/simonpintarelli/upf_to_json


Installation with pip
=====================

Using pip can easy install the release version of DFTpy from `PyPI <https://pypi.org/project/dftpy>`_::

    $ python -m pip install dftpy

Installation from source
========================

You can get the source from `github  <https://github.com/Quantum-MultiScale/DFTpy>`_.::

    $ git clone git@github.com:Quantum-MultiScale/DFTpy.git
    $ python -m pip install ./dftpy

Or in one line::
    
    $ python -m pip install git+git@github.com:Quantum-MultiScale/DFTpy.git


You also can install all the optional packages with::

    $ python -m pip install ./dftpy[all]


.. note::

    Because ``DFTpy`` still under active development, non-backward-compatible changes can happen at any time. Please, clone the lastest release often.
