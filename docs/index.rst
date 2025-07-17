PyMC-BART
===================================================
|Tests|
|Coverage|
|Black|


.. |Tests| image:: https://github.com/pymc-devs/pymc-bart/actions/workflows/test.yml/badge.svg
    :target: https://github.com/pymc-devs/pymc-bart

.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pymc-bart/branch/main/graph/badge.svg?token=ZqH0KCLKAE
    :target: https://codecov.io/gh/pymc-devs/pymc-bart

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



Bayesian Additive Regression Trees for Probabilistic programming with PyMC


Overview
============
PyMC-BART extends `PyMC <https://github.com/pymc-devs/pymc>`_ probabilistic programming framework to be able to define
and solve models including a BART random variable.  PyMC-BART also includes a few helpers function to aid with the
interpretation of those models and perform variable selection.


Installation
============

PyMC-BART requires a working Python interpreter (3.11+). We recommend installing Python and key numerical libraries using the `Anaconda distribution <https://www.anaconda.com/products/individual#Downloads>`_, which has one-click installers available on all major platforms.

Assuming a standard Python environment is installed on your machine, PyMC-BART itself can be installed either using pip or conda-forge.

**Using pip**

.. code-block:: bash

    pip install pymc-bart

**Using conda-forge**

.. code-block:: bash

     conda install -c conda-forge pymc-bart

**Development**

Alternatively, if you want the bleeding edge version of the package you can install from GitHub:

.. code-block:: bash

    pip install git+https://github.com/pymc-devs/pymc-bart.git


Citation
========
If you use PyMC-BART and want to cite it please use |arXiv|

.. |arXiv| image:: https://img.shields.io/badge/arXiv-2206.03619-b31b1b.svg
    :target: https://arxiv.org/abs/2206.03619

Here is the citation in BibTeX format

.. code-block::

    @misc{quiroga2022bart,
      doi = {10.48550/ARXIV.2206.03619},
      url = {https://arxiv.org/abs/2206.03619},
      author = {Quiroga, Miriana and Garay, Pablo G and Alonso, Juan M. and Loyola, Juan Martin and Martin, Osvaldo A},
      keywords = {Computation (stat.CO), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Bayesian additive regression trees for probabilistic programming},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }



Contributing
============
We welcome contributions from interested individuals or groups! For information about contributing to PyMC-BART check out our instructions, policies, and guidelines `here <https://github.com/pymc-devs/pymc-bart/blob/main/CONTRIBUTING.md>`_.

Contributors
============
See the `GitHub contributor page <https://github.com/pymc-devs/pymc-bart/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 2

   examples

References
==========

.. toctree::
   :maxdepth: 1

   api_reference
   changelog
