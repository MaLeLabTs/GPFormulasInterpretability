# Learning a Formula of Interpretability to Learn Interpretable Formulas

In this repository you can find the code and the supplementary material for our article about learning interpretable 
formulas with Genetic Programming.

You can find:

* the [supplementary](supplementary.pdf) material for the article
* the [datasets](datasets/) used for assessing our proposal
* the formulas used for the survey, both for [decomposability](dec1000-copy) and [simulatability](sim1000-copy)
* the [results](results_335.csv) of the survey
* the [code](Analisi.ipynb) used for analyzing the results of the survey, extract the features and compute the linear model
* the [results](gp-results/) of the experiments with pyNSGP

For the code of GP part, you can refer to [this](https://github.com/marcovirgolin/pyNSGP) repository.

### How to reproduce the experiments

In order to replicate our experiment, you can use the [run.py](run.py) script.
Such a script requires the datasets folder and takes as input:

1. the name of the dataset
1. the number of runs
1. (otpional) the model to use for interpretability. If you specify an 'm' the script execute experiments with out model, otherwise with the nodes count based model.


### Reference

If you use our code, please support our research by citing the related [paper](https://arxiv.org/abs/2004.11170):
> M. Virgolin, A. De Lorenzo, E. Medvet, F. Randone. "Learning a Formula of Interpretability to Learn Interpretable Formulas".  arXiv preprint arXiv:2004.11170 (2020)
