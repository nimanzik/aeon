from aeon.clustering import BaseClusterer

class Kmeans(BaseClusterer):

    _tags = {
        "capability:multivariate": True,
    }

    _init_algorithms = {
        "forgy": _forgy_center_initializer,
        "random": _random_center_initializer,
        "kmeans++": _kmeans_plus_plus,
    }

    def __init__(self, n_clusters: int = 8, init_algorithm: str = 'forgy', metric: str = 'dtw', n_init: int = 10, max_iter: int = 30, tol: float = 1e-6, verbose: bool = False, random_state: Optional[Union[int, np.random.RandomState]] = None, averaging_method: str = 'mean', average_params: Optional[Dict] = None, distance_params: Optional[Dict] = None):
        super().__init__(n_clusters=n_clusters, init_algorithm=init_algorithm, metric=metric, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose, random_state=random_state, averaging_method=averaging_method, average_params=average_params, distance_params=distance_params)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None) -> 'Kmeans':
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