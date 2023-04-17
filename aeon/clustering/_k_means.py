from typing import Optional, Union, Dict
from aeon.clustering import BaseClusterer
from aeon.clustering._init_algorithms import check_init_algorithm
from sklearn.utils import check_random_state
import numpy as np


class Kmeans(BaseClusterer):
    _tags = {
        "capability:multivariate": True,
    }

    def __init__(
            self,
            n_clusters: int = 8,
            init: Union[str, np.ndarray] = 'forgy',
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
        self._init = None

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

    def fit(self, X: np.ndarray, y: np.ndarray = None,
            sample_weight: np.ndarray = None) -> 'Kmeans':
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
        self._init = check_init_algorithm(self.init, self.n_clusters)

        best_centers = None
        best_interia = np.inf
        best_labels = None
        best_n_iter = None
        for _ in range(self.n_init):


        return self
