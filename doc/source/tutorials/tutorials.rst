.. _tutorials:

Tutorials
=========

Input file mode of DFTpy
----------------------

.. toctree::
   :maxdepth: 1

   config
   ofdft/optimize
   tddft/propagate

OFDFT
-----

.. toctree::
   :maxdepth: 1

   jupyter/density_optimization
   jupyter/relax
   jupyter/nvt

Local Pseudopotentials
----------------------

.. toctree::
   :maxdepth: 1

   jupyter/lpps

TDDFT
-----

.. toctree::
   :maxdepth: 1

   jupyter/td-ofdft-tutorial
   jupyter/lr-ofdft-tutorial

Do it on a Jupyter Notebook!
----------------------------

These notebooks are also built into this manual. To work on your machine, grab the ``.ipynb`` from
`GitHub <https://github.com/Quantum-MultiScale/DFTpy>`__ (**⋯ → Download** on the file page). To try them in **Google Colab**, use the Colab links below. Install DFTpy with ``pip`` (often ``pip install dftpy`` or ``pip install 'dftpy[...]'`` per the project). 

.. For **LibXC + pylibxc** (e.g. Casida with LibXC functionals), Colab is Ubuntu-based: install the system development package so the C library and headers exist, then install the Python bindings:

.. code-block:: python

   # Run in Colab before importing the DFTpy modules
   !pip install dftpy

* **Density optimization** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/density_optimization.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/density_optimization.ipynb>`__
* **NVT molecular dynamics (ASE)** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/nvt.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/nvt.ipynb>`__
* **LPP optimization (LPPS)** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/dev/examples/Pseudopotentials/lpps.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/dev/examples/Pseudopotentials/lpps.ipynb>`__
* **Real-time TD-OFDFT** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/td-ofdft-tutorial.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/td-ofdft-tutorial.ipynb>`__
* **Linear-response TD-OFDFT (Casida)** — `GitHub <https://github.com/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/lr-ofdft-tutorial.ipynb>`__ · `Colab <https://colab.research.google.com/github/Quantum-MultiScale/DFTpy/blob/dev/examples/notebooks/lr-ofdft-tutorial.ipynb>`__

.. toctree::
   :hidden:

   jupyter/density_optimization

