# To Impute or not to Impute? Missing Data in Treatment Effect Estimation </br><sub><sub>J. Berrevoets, F. Imrie, T. Kyono, J. Jordon, M. van der Schaar [[AISTATS 2023]](https://arxiv.org/abs/2202.02096)</sub></sub>

<div align="center">

[![arXiv](https://img.shields.io/badge/paper-AISTATS2023-orange)](https://arxiv.org/abs/2202.02096)
[![Experiments](https://github.com/vanderschaarlab/mcm/actions/workflows/test_experiments.yml/badge.svg)](https://github.com/vanderschaarlab/mcm/actions/workflows/test_experiments.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vanderschaarlab/mcm/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![about](https://img.shields.io/badge/about-The%20van%20der%20Schaar%20Lab-blue)](https://www.vanderschaar-lab.com/)

</div>


![Screenshot 2023-02-14 at 14 49 20](https://user-images.githubusercontent.com/6019254/218757241-8058b6b7-9263-407a-83fc-9b6d8f60923b.png)

In this repository we provide code for our AISTATS23 paper introducing MCM, a novel missingness mechanism for treatment effect inference. Note that this code is used for research purposes and is __not intented for use in practice__.

_Code author: J. Berrevoets ([jb2384@cam.ac.uk](mailto:jb2384@cam.ac.uk))_

## Installation

```bash
pip install -r requirements.txt
```

## Repository structure
This repository is organised as follows:
```bash
mcm/
    |- src/
        |- data/
            |- data_module.py               # code to simulate MCM data
            |- utils.py                     # code to split data
    |- notebooks/
        |- <experiment>.ipynb               # dedicated notebook for experiment
        |- simple_setup.ipynb               # self contained notebook with basic experiment
```


Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Citing
If you use this code, please cite the associated paper:

```
@misc{https://doi.org/10.48550/arxiv.2202.02096,
  doi = {10.48550/ARXIV.2202.02096},
  url = {https://arxiv.org/abs/2202.02096},
  author = {Berrevoets, Jeroen and Imrie, Fergus and Kyono, Trent and Jordon, James and van der Schaar, Mihaela},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {To Impute or not to Impute? Missing Data in Treatment Effect Estimation},
}
```
