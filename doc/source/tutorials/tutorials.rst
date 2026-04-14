.. _tutorials:

Tutorials
=========

Config of DFTpy script
----------------------

.. toctree::
   :maxdepth: 1

   config

OFDFT
-----

.. toctree::
   :maxdepth: 1

   ofdft/optimize
   ofdft/relax
   ofdft/md

Local Pseudopotentials
----------------------

.. toctree::
   :maxdepth: 1

   jupyter/lpps

TDDFT
-----

.. toctree::
   :maxdepth: 1

   tddft/propagate
   jupyter/td-ofdft-tutorial
   jupyter/lr-ofdft-tutorial

Do it on a Jupyter Notebook!
----------------------------

These notebooks are also built into this manual. To work on your machine, grab the ``.ipynb`` from
`GitHub <https://github.com/Quantum-MultiScale/DFTpy>`__ (**⋯ → Download** on the file page). To try them in **Google Colab**, use the Colab links below. Install DFTpy with ``pip`` (often ``pip install dftpy`` or ``pip install 'dftpy[...]'`` per the project). For **LibXC + pylibxc** (e.g. Casida with LibXC functionals), Colab is Ubuntu-based: install the system development package so the C library and headers exist, then install the Python bindings:

.. code-block:: python

   # Run in Colab before importing Casida / LibXC-dependent code
   !apt-get update -qq && apt-get install -qq libxc-dev
   !pip install -q pylibxc

If ``pip install pylibxc`` fails (no wheel for that Colab image), build ``pylibxc`` from the `LibXC <https://gitlab.com/libxc/libxc>`_ source tree after ``libxc-dev`` is installed, or use a conda env with ``libxc`` / ``pylibxc`` instead. Other optional deps: ``pip install h5py`` for HDF5 examples.

* **Density optimization** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/density_optimization.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/density_optimization.ipynb>`__
* **LPP optimization (LPPS)** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/master/examples/Pseudopotentials/lpps.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/master/examples/Pseudopotentials/lpps.ipynb>`__
* **Real-time TD-OFDFT** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/td-ofdft-tutorial.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/td-ofdft-tutorial.ipynb>`__
* **Linear-response TD-OFDFT (Casida)** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/lr-ofdft-tutorial.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/master/examples/notebooks/lr-ofdft-tutorial.ipynb>`__

.. toctree::
   :hidden:

   jupyter/density_optimization

