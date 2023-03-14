# -*- coding: utf-8 -*-
"""Test functions for loose loaers."""

__author__ = ["fkiraly"]

__all__ = []


import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_load_macroeconomic():
    """Test that load_macroeconomic runs."""
    from aeon.datasets import load_macroeconomic

    load_macroeconomic()