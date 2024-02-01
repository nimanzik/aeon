"""Base classes for defining estimators and other objects in aeon."""

__author__ = ["mloning", "RNKuhns", "fkiraly", "TonyBagnall"]
__all__ = [
    "BaseObject",
    "BaseEstimator",
    "BaseCollectionEstimator",
    "_HeterogenousMetaEstimator",
    "load",
    "COLLECTIONS_DATA_TYPES",
]

from aeon.base._base import BaseEstimator, BaseObject
from aeon.base._base_collection import COLLECTIONS_DATA_TYPES, BaseCollectionEstimator
from aeon.base._meta import _HeterogenousMetaEstimator
from aeon.base._serialize import load
