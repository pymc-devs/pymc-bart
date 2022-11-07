
Bayesian Additive Regression Trees for Probabilistic programming with PyMC


PyMC-BART extends [PyMC](https://github.com/pymc-devs/pymc) probabilistic programming framework to be able to define and solve models including a BART random variable. PyMC-BART also includes a few helpers function to aid with the interpretation of those models and perform variable selection. 


## Installation

PyMC-BART is available on Conda-Forge. To set up a suitable Conda environment, run

```bash
conda create --name=pymc-bart --channel=conda-forge pymc-bart
conda activate pymc-bart
```

Alternatively, it can be installed with

```bash
pip install pymc-bart
```

In case you want to upgrade to the bleeding edge version of the package you can install from GitHub:

```bash
pip install git+https://github.com/pymc-devs/pymc-bart.git
```

## Contributions
PyMC-BART is a community project and welcomes contributions.
Additional information can be found in the [Contributing Readme](https://github.com/pymc-devs/pymc_bart/blob/main/CONTRIBUTING.md)

## Code of Conduct
PyMC-BART wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://github.com/pymc-devs/pymc_bart/blob/main/CODE_OF_CONDUCT.md)

## Citation
If you use PyMC-BART and want to cite it please use [![arXiv](https://img.shields.io/badge/arXiv-2206.03619-b31b1b.svg)](https://arxiv.org/abs/2206.03619)

Here is the citation in BibTeX format

```
@misc{quiroga2022bart,
title={Bayesian additive regression trees for probabilistic programming},
author={Quiroga, Miriana and Garay, Pablo G and Alonso, Juan M. and Loyola, Juan Martin and Martin, Osvaldo A},
year={2022},
doi={10.48550/ARXIV.2206.03619},
archivePrefix={arXiv},
primaryClass={stat.CO}
}
```

## Donations
PyMC-BART , as other pymc-devs projects, is a non-profit project under the NumFOCUS umbrella. If you want to support PyMC-BART financially, you can donate [here](https://numfocus.org/donate-to-pymc).

## Sponsors
[![NumFOCUS](https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png)](https://numfocus.org)
