# -*- coding: utf-8 -*-
"""Distance module."""
__all__ = [
    "create_bounding_matrix",
    "euclidean_distance",
    "squared_distance",
    "dtw_distance",
    "dtw_cost_matrix",
    "ddtw_distance",
    "ddtw_cost_matrix",
    "wdtw_distance",
    "wdtw_cost_matrix",
    "wddtw_distance",
    "wddtw_cost_matrix",
    "edr_distance",
    "edr_cost_matrix",
    "edr_alignment_path",
    "erp_distance",
    "erp_cost_matrix",
    "erp_alignment_path",
    "lcss_distance",
    "lcss_cost_matrix",
    "msm_distance",
    "msm_cost_matrix",
    "twe_distance",
    "twe_cost_matrix",
    "dtw_pairwise_distance",
    "dtw_from_multiple_to_multiple_distance",
    "dtw_from_single_to_multiple_distance",
    "euclidean_pairwise_distance",
    "euclidean_from_single_to_multiple_distance",
    "euclidean_from_multiple_to_multiple_distance",
    "squared_pairwise_distance",
    "squared_from_single_to_multiple_distance",
    "squared_from_multiple_to_multiple_distance",
    "ddtw_pairwise_distance",
    "ddtw_from_multiple_to_multiple_distance",
    "ddtw_from_single_to_multiple_distance",
    "wdtw_pairwise_distance",
    "wdtw_from_multiple_to_multiple_distance",
    "wdtw_from_single_to_multiple_distance",
    "wdtw_alignment_path",
    "wddtw_pairwise_distance",
    "wddtw_from_multiple_to_multiple_distance",
    "wddtw_from_single_to_multiple_distance",
    "wddtw_alignment_path",
    "edr_pairwise_distance",
    "edr_from_multiple_to_multiple_distance",
    "edr_from_single_to_multiple_distance",
    "erp_pairwise_distance",
    "erp_from_multiple_to_multiple_distance",
    "erp_from_single_to_multiple_distance",
    "lcss_pairwise_distance",
    "lcss_from_multiple_to_multiple_distance",
    "lcss_from_single_to_multiple_distance",
    "lcss_alignment_path",
    "msm_pairwise_distance",
    "msm_from_multiple_to_multiple_distance",
    "msm_from_single_to_multiple_distance",
    "msm_alignment_path",
    "twe_pairwise_distance",
    "twe_from_multiple_to_multiple_distance",
    "twe_from_single_to_multiple_distance",
    "twe_alignment_path",
    "distance",
    "pairwise_distance",
    "distance_from_single_to_multiple",
    "distance_from_multiple_to_multiple",
    "cost_matrix",
    "ddtw_alignment_path",
    "dtw_alignment_path",
    "alignment_path",
    "distance_function_dict",
    "cost_matrix_function_dict",
    "single_to_multiple_distance_function_dict",
    "pairwise_distance_function_dict",
    "alignment_path_function_dict",
    "multiple_to_multiple_distance_function_dict",
    "compute_min_return_path",
    "compute_lcss_return_path",
]

from aeon.distances._alignment_paths import (
    compute_lcss_return_path,
    compute_min_return_path,
)
from aeon.distances._bounding_matrix import create_bounding_matrix
from aeon.distances._ddtw import (
    ddtw_alignment_path,
    ddtw_cost_matrix,
    ddtw_distance,
    ddtw_from_multiple_to_multiple_distance,
    ddtw_from_single_to_multiple_distance,
    ddtw_pairwise_distance,
)
from aeon.distances._distance import (
    alignment_path,
    alignment_path_function_dict,
    cost_matrix,
    cost_matrix_function_dict,
    distance,
    distance_from_multiple_to_multiple,
    distance_from_single_to_multiple,
    distance_function_dict,
    multiple_to_multiple_distance_function_dict,
    pairwise_distance,
    pairwise_distance_function_dict,
    single_to_multiple_distance_function_dict,
)
from aeon.distances._dtw import (
    dtw_alignment_path,
    dtw_cost_matrix,
    dtw_distance,
    dtw_from_multiple_to_multiple_distance,
    dtw_from_single_to_multiple_distance,
    dtw_pairwise_distance,
)
from aeon.distances._edr import (
    edr_alignment_path,
    edr_cost_matrix,
    edr_distance,
    edr_from_multiple_to_multiple_distance,
    edr_from_single_to_multiple_distance,
    edr_pairwise_distance,
)
from aeon.distances._erp import (
    erp_alignment_path,
    erp_cost_matrix,
    erp_distance,
    erp_from_multiple_to_multiple_distance,
    erp_from_single_to_multiple_distance,
    erp_pairwise_distance,
)
from aeon.distances._euclidean import (
    euclidean_distance,
    euclidean_from_multiple_to_multiple_distance,
    euclidean_from_single_to_multiple_distance,
    euclidean_pairwise_distance,
)
from aeon.distances._lcss import (
    lcss_alignment_path,
    lcss_cost_matrix,
    lcss_distance,
    lcss_from_multiple_to_multiple_distance,
    lcss_from_single_to_multiple_distance,
    lcss_pairwise_distance,
)
from aeon.distances._msm import (
    msm_alignment_path,
    msm_cost_matrix,
    msm_distance,
    msm_from_multiple_to_multiple_distance,
    msm_from_single_to_multiple_distance,
    msm_pairwise_distance,
)
from aeon.distances._squared import (
    squared_distance,
    squared_from_multiple_to_multiple_distance,
    squared_from_single_to_multiple_distance,
    squared_pairwise_distance,
)
from aeon.distances._twe import (
    twe_alignment_path,
    twe_cost_matrix,
    twe_distance,
    twe_from_multiple_to_multiple_distance,
    twe_from_single_to_multiple_distance,
    twe_pairwise_distance,
)
from aeon.distances._wddtw import (
    wddtw_alignment_path,
    wddtw_cost_matrix,
    wddtw_distance,
    wddtw_from_multiple_to_multiple_distance,
    wddtw_from_single_to_multiple_distance,
    wddtw_pairwise_distance,
)
from aeon.distances._wdtw import (
    wdtw_alignment_path,
    wdtw_cost_matrix,
    wdtw_distance,
    wdtw_from_multiple_to_multiple_distance,
    wdtw_from_single_to_multiple_distance,
    wdtw_pairwise_distance,
)
