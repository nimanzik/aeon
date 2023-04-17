# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]

from typing import Callable, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils.extmath import stable_cumsum

from aeon.clustering.metrics.averaging import mean_average
from aeon.distances import distance_from_multiple_to_multiple

InitCallable = Callable[[np.ndarray, int, RandomState, dict], np.ndarray]
CenterComputeCallable = Callable[[np.ndarray, dict], np.ndarray]

init_algorithm_dict = {
    "forgy": "forgy_center_initializer",
    "random": "random_center_initializer",
    "kmeans++": "kmeans_plus_plus",
}


def check_init_algorithm(
        init_algorithm: Union[str, np.ndarray, InitCallable], n_clusters: int
) -> InitCallable:
    """Check the initialization algorithm.

    If an array is passed in then it is wrapped in a function to be consistent with
    the return type.

    Parameters
    ----------
    init_algorithm: Union[str, np.ndarray, Callable]
        The initialization algorithm to use. If a string, must be one of
        'forgy', 'random', or 'kmeans++'. If an array, must be of shape
        (n_clusters, n_channels, n_timepoints). If a callable, must take
        the form Callable[[np.ndarray, int, np.random.RandomState, dict], np.ndarray].

    n_clusters: int
        The number of clusters to initialize.

    Returns
    -------
    Callable[[np.ndarray, int, np.random.RandomState, dict], np.ndarray]
        An initialization callable
    """
    if isinstance(init_algorithm, str):
        if init_algorithm not in init_algorithm_dict:
            raise ValueError(
                f"Unknown initialization algorithm '{init_algorithm}'."
                f"Valid options are {list(init_algorithm_dict.keys())}"
            )
        return init_algorithm_dict[init_algorithm]
    elif isinstance(init_algorithm, np.ndarray):
        if init_algorithm.shape[0] != n_clusters:
            raise ValueError(
                f"Number of clusters ({n_clusters}) does not match the number of "
                f"initial centers ({init_algorithm.shape[0]})."
            )

        def _inner_init_algorithm(_X, _n_clusters, _random_state):
            return init_algorithm

        return _inner_init_algorithm
    elif callable(init_algorithm):
        return init_algorithm
    else:
        raise ValueError(
            f"Unknown initialization algorithm '{init_algorithm}'."
            f"Valid options are {list(init_algorithm_dict.keys())}"
        )


def forgy_center_initializer(
        X: np.ndarray, n_clusters: int, random_state: np.random.RandomState, **kwargs
) -> np.ndarray:
    """Compute the initial centers using forgy method.

    Forgy works by randomly selecting n_clusters from the dataset.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances, n_channels, series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_channels, series_length))
        Time series instances that are the cluster centers.
    """
    return X[random_state.choice(X.shape[0], n_clusters, replace=False)]


def random_center_initializer(
        X: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        center_compute_method: CenterComputeCallable = mean_average,
        center_compute_kwargs: dict = None,
        **kwargs
) -> np.ndarray:
    """Compute initial centroids using random method.

    This works by assigning each point randomly to a cluster. Then the average of
    the cluster is taken to get the centers.

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances, n_channels, series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.
    center_compute_method: Callable[[np.ndarray, dict], np.ndarray],
            defaults = mean_average
        The method to use to compute the center of a cluster. The method is called
        by center_compute_method(X[curr_indexes], **center_compute_kwargs).
    center_compute_kwargs: dict, defaults = None
        The keyword arguments to pass to the center_compute_method.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_channels, series_length))
        Time series instances that are the cluster centers.
    """
    if center_compute_kwargs is None:
        center_compute_kwargs = {}
    new_centres = np.zeros((n_clusters, X.shape[1], X.shape[2]))
    selected = random_state.choice(n_clusters, X.shape[0], replace=True)
    for i in range(n_clusters):
        curr_indexes = np.where(selected == i)[0]
        result = center_compute_method(X[curr_indexes], **center_compute_kwargs)
        if result.shape[0] > 0:
            new_centres[i, :] = result

    return new_centres


def kmeans_plus_plus(
        X: np.ndarray,
        n_clusters: int,
        random_state: np.random.RandomState,
        distance_metric: str = "euclidean",
        n_local_trials: int = None,
        distance_params: dict = None,
        **kwargs
):
    """Compute initial centroids using kmeans++ method.

    This works by choosing one point at random. Next compute the distance between the
    center and each point. Sample these with a probability proportional to the square
    of the distance of the points from its nearest center.

    NOTE: This is adapted from sklearns implementation:
    https://
    github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/cluster/_kmeans.py

    Parameters
    ----------
    X : np.ndarray (3d array of shape (n_instances,n_dimensions,series_length))
        Time series instances to cluster.
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    random_state: np.random.RandomState
        Determines random number generation for centroid initialization.
    distance_metric: str, defaults = 'euclidean'
        String that is the distance metric.
    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    distance_params: dict, defaults = None
        Dictionary containing distance parameter kwargs.

    Returns
    -------
    np.ndarray (3d array of shape (n_clusters, n_dimensions, series_length))
        Indexes of the cluster centers.
    """
    n_samples, n_timestamps, n_features = X.shape

    centers = np.empty((n_clusters, n_timestamps, n_features), dtype=X.dtype)
    n_samples, n_timestamps, n_features = X.shape

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    if distance_params is None:
        distance_params = {}

    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]
    closest_dist_sq = (
            distance_from_multiple_to_multiple(
                centers[0, np.newaxis], X, metric=distance_metric
            )
            ** 2
    )
    current_pot = closest_dist_sq.sum()

    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        distance_to_candidates = (
                distance_from_multiple_to_multiple(
                    X[candidate_ids], X, metric=distance_metric, **distance_params
                )
                ** 2
        )

        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        centers[c] = X[best_candidate]

    return centers
