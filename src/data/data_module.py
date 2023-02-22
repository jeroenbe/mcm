# third party
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy


def generate_data_exp(
    n: int,
    d: int,
    z_d_dim: int,
    amount_of_missingness: float,
    missing_value: float = -1,
    data: str = "synth",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # GENERATE DATA

    assert 0 < n
    assert 0 < z_d_dim <= d
    assert (
        0 < amount_of_missingness <= 0.5
    )  # Note that this is approximate, sampling is still random

    # X
    if data == "synth":
        X = _generate_covariates(d, n)  # Contains negatives. We will use negative
        # values to detirmine some of the data, only
        # to remove them later, resulting in basic
        # non-linearities (identifiability in the DAG).
    elif data == "twins":
        X = _get_twins_data_covariates(n=n)

    # OUTCOMES
    Y0, Y1, CATE = _generate_outcomes(X)

    # DOWN
    Z_down = _Z_down(amount_of_missingness, X, z_d_dim)

    # TREATMENTS
    W = _treatments(Z_down, X, z_d_dim)
    # OBSERVED Y
    Y = _generate_observed_outcomes(Y0, Y1, W)

    # UP
    Z_up = _Z_up(amount_of_missingness, X, z_d_dim, W)

    # COMPLETE DATA
    X_ = _complete_covariates(X, z_d_dim, Z_up, Z_down, missing_value)

    return X, X_, Y0, Y1, Y, CATE, W, Z_up, Z_down


def _generate_covariates(d: int, n: int) -> np.ndarray:
    assert 0 < d
    assert 0 < n

    # COVARIATES
    # X = np.random.rand(n, drandom.multivariate_normal)         # Fully observed X
    A = np.random.rand(d, d)
    cov = np.dot(A, A.transpose())

    X = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    X /= X.max() - X.min()

    return X


def _get_twins_data_covariates(
    location: str = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS",
    n: Optional[int] = None,
) -> np.ndarray:

    X = pd.read_csv(f"{location}/twin_pairs_X_3years_samesex.csv")
    X = X.dropna(axis=0)

    lighter_columns = [
        "pldel",
        "birattnd",
        "brstate",
        "stoccfipb",
        "mager8",
        "ormoth",
        "mrace",
        "meduc6",
        "dmar",
        "mplbir",
        "mpre5",
        "adequacy",
        "orfath",
        "frace",
        "birmon",
        "gestat10",
        "csex",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "tobacco",
        "alcohol",
        "cigar6",
        "drink5",
        "crace",
        "data_year",
        "nprevistq",
        "dfageq",
        "feduc6",
        "infant_id_0",
        "dlivord_min",
        "dtotord_min",
        "bord_0",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
    ]

    lighter = X[lighter_columns]

    X = lighter  # .append(heavier)
    X = X.dropna(axis=1)

    if n:
        X.sample(n, replace=False)

    X = X.to_numpy()

    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X


def _generate_outcomes(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.random.randn(X.shape[1]) / 10
    theta_y0 = np.ones(X.shape[1]) + theta
    theta_y1 = np.ones(X.shape[1]) * -1 + theta

    Y0 = np.sum(X * theta_y0, 1)
    Y1 = np.sum(X * theta_y1, 1)

    CATE = Y1 - Y0

    return Y0, Y1, CATE


def _generate_observed_outcomes(
    Y0: np.ndarray, Y1: np.ndarray, W: np.ndarray
) -> np.ndarray:
    return (
        np.array([Y0[i] if w == 0 else Y1[i] for i, w in enumerate(W)])
        + np.random.randn(W.shape[0]) * 0.1
    )


def _Z_down(amount_of_missingness: float, X: np.ndarray, z_d_dim: int) -> np.ndarray:
    highest_border = X[:, :z_d_dim].argsort(axis=1)[
        :, -int(np.max((int(np.round(amount_of_missingness * z_d_dim)), 1)))
    ]
    Z_down = np.array(
        list(x >= x[highest_border[i]] for i, x in enumerate(X[:, :z_d_dim]))
    ).astype(int)
    return np.abs(Z_down - 1)  # 0 = missing, 1 = present


def _treatments(Z_down: np.ndarray, X: np.ndarray, z_d_dim: int) -> np.ndarray:
    W = []
    for z_d in Z_down:
        if 0 == z_d[-1]:
            w = 0
        elif 0 in z_d[: int(np.floor(z_d_dim / 2))]:
            w = 1
        else:
            w = np.random.binomial(1, 0.5)
        W.append(w)
    return np.array(W)


def _Z_up(
    amount_of_missingness: float, X: np.ndarray, z_d_dim: int, W: np.ndarray
) -> np.ndarray:
    d = X.shape[1]
    dim_count = np.round(amount_of_missingness * (d - z_d_dim) * 2)
    dim_count = np.max((dim_count, 1))
    dim_count = np.min((dim_count, int((d - z_d_dim) / 2)))
    dim_count = int(dim_count)

    theta_z_in_0 = np.random.normal(
        loc=scipy.stats.norm.ppf(
            1 - amount_of_missingness
        ),  # We translate the binary treatment
        size=dim_count,
        scale=0.5,
    )  # to a more complex interaction. Recall
    theta_z_in_1 = np.random.normal(  # that each arrow needs to be identifiable.
        loc=scipy.stats.norm.ppf(
            1 - amount_of_missingness
        ),  # Solely relying on the binary information
        size=dim_count,
        scale=0.5,
    )  # makes this very hard, if not impossible.

    theta_z_in_0 = np.full(dim_count, scipy.stats.norm.ppf(1 - amount_of_missingness))
    theta_z_in_1 = np.full(dim_count, scipy.stats.norm.ppf(1 - amount_of_missingness))

    n = X.shape[0]

    Z_up = np.zeros((n, d - z_d_dim))
    for i, z in enumerate(Z_up):
        x = X[
            i, z_d_dim : z_d_dim + dim_count
        ]  # Again, relying on negatives in X will help
        # make each dependency identifiable.
        if W[i]:
            Z_up[i, -dim_count:] = (
                x - X[:, z_d_dim : z_d_dim + dim_count].mean(axis=0)
            ) > (
                theta_z_in_1 * x.std(axis=0)
            )  # if W[i] else x > (-1 * theta_z_in)
        else:
            Z_up[i, :dim_count] = (
                x - X[:, z_d_dim : z_d_dim + dim_count].mean(axis=0)
            ) > (
                theta_z_in_0 * x.std(axis=0)
            )  # WAS :dim_count instead of -dim_count: -> trying out MAR throughout

    Z_up = np.abs(Z_up - 1)

    return Z_up


def _complete_covariates(
    X: np.ndarray,
    z_d_dim: int,
    Z_up: np.ndarray,
    Z_down: np.ndarray,
    missing_value: float,
) -> np.ndarray:

    X = np.abs(X)  # the non-linearity for identifiability of our DAG

    # X_tilde
    X_ = X.copy()
    X_[:, z_d_dim:][Z_up == 0] = missing_value
    X_[:, :z_d_dim][Z_down == 0] = missing_value

    return X_
