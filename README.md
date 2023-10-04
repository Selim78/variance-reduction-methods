# Variance Reduction Methods

Partial re-implementation of [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (using only numpy) to illustrate the use of variance reduction methods in stochastic optimization.

![](https://i.imgur.com/pA8hGhZ.png)

## Contents

* A small [report](https://SelimChraibi.github.io/variance-reduction-methods/) on the intuition behind stochastic variance reduction in optimisation & how to use the code.
[![Report.html](https://i.imgur.com/9EMkAMh.png)](https://SelimChraibi.github.io/variance-reduction-methods/)

* [`Report.ipynb`](https://nbviewer.jupyter.org/github/SelimChraibi/variance-reduction-methods/blob/master/Report.ipynb): same as the html report, in case you want to reproduce the results
* Implementation broken down into:
    * `linear_model.py`
    * `solvers.py`
    * Helper functions: `datasets.py`, `visuals.py`, `tools.py`

    * [Student performance](http://archive.ics.uci.edu/ml/datasets/Student+Performance) dataset: `data/`

## Reproducing the results

To get started with the [`Report.ipynb`](https://nbviewer.jupyter.org/github/SelimChraibi/variance-reduction-methods/blob/master/Report.ipynb) notebook, create an environment using the dependencies file:

```bash
conda env create --file dependencies.yml
```

Then launch `jupyter-notebook` and select `Kernel -> Change kernel -> Python [conda env:vrm]`
