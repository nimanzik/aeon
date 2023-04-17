from typing import Callable
import numpy as np
from aeon.clustering._init_algorithms import (
    forgy_center_initializer, random_center_initializer, kmeans_plus_plus
)


def _test_cluster_centers_init(callable: Callable, random_state: np.random.RandomState):
    num_instances = [10, 20, 30]
    num_dims = [1, 10]
    num_timepoints = [10, 20, 30]
    num_clusters = [1, 5, 10]
    for n_clusters in num_clusters:
        for n in num_instances:
            for d in num_dims:
                for t in num_timepoints:
                    X = np.random.rand(n, d, t)
                    centers = callable(X, n_clusters, random_state)
                    assert centers.shape == (n_clusters, X.shape[1], X.shape[2])


def test_forgy():
    _test_cluster_centers_init(forgy_center_initializer, np.random.RandomState(0))


def test_random():
    _test_cluster_centers_init(random_center_initializer, np.random.RandomState(0))


def test_kmeans_plus_plus():
    _test_cluster_centers_init(kmeans_plus_plus, np.random.RandomState(0))
