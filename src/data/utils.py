# third party
from typing import Tuple

import numpy as np


def split_eval(
    results: np.ndarray, truth: np.ndarray, split_amount: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    results = np.array(results).flatten()  # We evaluate results per fixed simulation
    truth = np.array(truth).flatten()  # and then average once computed. This is a
    # way to cope with the variance of the simulation.
    error = (results - truth) ** 2  # Note that this makes no difference for the mean
    error = np.array(
        np.split(error, split_amount)
    )  # (as the mean of subset-means is still the mean),
    # but it makes a difference for calculating the std.
    return error.mean(axis=1), error.std(
        axis=1
    )  # If we do not calculate the std according to this,
    # then the given std will be biased towards the
    # variance of the simulation's distribution.


def split_eval_cate(
    results: np.ndarray, truth: np.ndarray, split_amount: float
) -> Tuple[np.ndarray, np.ndarray]:  # Same logic here as above.

    error = np.array([((l - truth[i]) ** 2).mean() for i, l in enumerate(results)])
    error = np.array(np.split(error, split_amount))

    return error.mean(axis=1), error.std(axis=1)
