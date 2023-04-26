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

@InProceedings{pmlr-v206-berrevoets23a,
  title = 	 {To Impute or not to Impute? Missing Data in Treatment Effect Estimation},
  author =       {Berrevoets, Jeroen and Imrie, Fergus and Kyono, Trent and Jordon, James and van der Schaar, Mihaela},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3568--3590},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/berrevoets23a/berrevoets23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/berrevoets23a.html},
  abstract = 	 {Missing data is a systemic problem in practical scenarios that causes noise and bias when estimating treatment effects. This makes treatment effect estimation from data with missingness a particularly tricky endeavour. A key reason for this is that standard assumptions on missingness are rendered insufficient due to the presence of an additional variable, treatment, besides the input (e.g. an individual) and the label (e.g. an outcome). The treatment variable introduces additional complexity with respect to why some variables are missing that is not fully explored by previous work. In our work we introduce mixed confounded missingness (MCM), a new missingness mechanism where some missingness determines treatment selection and other missingness is determined by treatment selection. Given MCM, we show that naively imputing all data leads to poor performing treatment effects models, as the act of imputation effectively removes information necessary to provide unbiased estimates. However, no imputation at all also leads to biased estimates, as missingness determined by treatment introduces bias in covariates. Our solution is selective imputation, where we use insights from MCM to inform precisely which variables should be imputed and which should not. We empirically demonstrate how various learners benefit from selective imputation compared to other solutions for missing data. We highlight that our experiments encompass both average treatment effects and conditional average treatment effects.}
}

```
