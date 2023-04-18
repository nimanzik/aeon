from typing import Optional, Union, Tuple
from numpy.random import RandomState
from sklearn.utils import check_random_state
import numpy as np

from aeon.clustering import BaseClusterer
from aeon.clustering._init_algorithms import InitCallable, check_init_algorithm
from aeon.distances import distance_from_multiple_to_multiple
from aeon.clustering.base import TimeSeriesInstances


class Kmeans(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
            self,
            n_clusters: int = 8,
            init: Union[str, np.ndarray, InitCallable] = 'forgy',
            metric: str = 'dtw',
            n_init: int = 10,
            max_iter: int = 30,
            tol: float = 1e-6,
            verbose: bool = False,
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            averaging_method: str = 'mean',
            init_params: dict = None,
            average_params: dict = None,
            distance_params: dict = None,
    ):
        self.init = init
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.distance_params = distance_params
        self.averaging_method = averaging_method
        self.init_params = init_params
        self.average_params = average_params

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        self._random_state = None
        self._average_params = average_params
        if average_params is None:
            self._average_params = {}
        self._distance_params = distance_params
        if distance_params is None:
            self._distance_params = {}
        self._init_params = init_params
        if init_params is None:
            self._init_params = {}

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        super(Kmeans, self).__init__(n_clusters=n_clusters)

    def _fit(self, X: TimeSeriesInstances, y=None) -> 'Kmeans':
        """Compute k-means clustering.

        Parameters
        ----------
        X: np.ndarray (3d array of shape (n_instances, n_dimensions, series_length))
            Time series to cluster.
        y: np.ndarray (1d array of shape (n_instances,), defaults = None)
            Ignored.
        sample_weight: np.ndarray (1d array of shape (n_instances,), defaults = None)
            The weights for each observation in X. If None, all observations are
            assigned equal weight (default: None).

        Returns
        -------
        self: Kmeans
            Fitted estimator.
        """
        self._random_state = check_random_state(self.random_state)
        init_callable = check_init_algorithm(self.init, self.n_clusters)

        best_centers = None
        best_interia = np.inf
        best_labels = None
        best_n_iter = None
        for _ in range(self.n_init):
            centers = init_callable(
                X, self.n_clusters, self._random_state, self._init_params
            )
            labels, centers, inertia, n_iter = self._fit_one_init(
                X,
                centers
            )
            if inertia < best_interia:
                best_centers = centers
                best_interia = inertia
                best_labels = labels
                best_n_iter = n_iter

        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_interia
        self.n_iter_ = best_n_iter
        return self

    def _score(self, X, y=None) -> float:
        return -self.inertia_

    def _predict(self, X: TimeSeriesInstances, y=None) -> np.ndarray:
        return self._assign_values_to_clusters(X, self.cluster_centers_)[0]

    def _fit_one_init(
            self,
            X: np.ndarray,
            cluster_centers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Performs one pass of kmeans

        Parameters
        ----------
        X: np.ndarray (n_instances, n_channels, n_timepoints)
            A collection of time series instances.
        cluster_centers: np.ndarray (n_clusters, n_channels, n_timepoints)
            The initial cluster centers.

        Returns
        -------
        labels: np.ndarray (n_instances,)
            The cluster labels for each instance.
        cluster_centers: np.ndarray (n_clusters, n_channels, n_timepoints)
            The cluster centers the algorithm converged at.
        inertia: float
            The sum of squared distances between each instance and its closest
            cluster center.
        n_iter: int
            The number of iterations the algorithm ran for before converging.
        """
        previous_inertia = np.inf
        previous_labels = None
        inertia = np.inf
        labels = None
        i = 0
        while i < self.max_iter:
            i += 1
            labels, inertia = self._assign_values_to_clusters(X, cluster_centers)
            if self.verbose:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001, T201

            if np.abs(previous_inertia - inertia) < self.tol:
                if self.verbose:
                    print(
                        f"Converged at iteration {i}: inertia"
                        f"{inertia} within tolerance {self.tol}."  # noqa: T001, T201
                    )
                break
            if previous_labels is not None and np.array_equal(labels, previous_labels):
                if self.verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                break
            previous_inertia = inertia
            previous_labels = labels

            cluster_centers = self._compute_new_cluster_centres(X, labels)

            if self.verbose is True:
                print(f"Iteration {i}, inertia {inertia}.")  # noqa: T001, T201

        return labels, cluster_centers, inertia, i

    def _assign_values_to_clusters(
            self, X: np.ndarray, cluster_centers: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        distance_to_centers = distance_from_multiple_to_multiple(
            X, cluster_centers, metric=self.metric, **self.distance_params
        )
        return distance_to_centers.argmin(axis=1), distance_to_centers.min(axis=1).sum()

    def _compute_new_cluster_centres(
            self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        return X[0:self.n_clusters]
        # return average_multiple(
        #     X, labels, method=self.averaging_method, **self._average_params
        # )
