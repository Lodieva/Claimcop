"""
Unit tests voor de pure, side-effect-vrije functies in src/inference.py.
Deze tests laden GEEN model en hebben GEEN netwerktoegang nodig, zodat ze
snel en betrouwbaar in CI kunnen draaien.

Run: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import mask_area_percentage


def test_mask_area_percentage_empty_mask():
    mask = np.zeros((100, 100))
    assert mask_area_percentage(mask) == 0.0


def test_mask_area_percentage_full_mask():
    mask = np.ones((100, 100))
    assert mask_area_percentage(mask) == 100.0


def test_mask_area_percentage_half_mask():
    mask = np.zeros((10, 10))
    mask[:5, :] = 1.0  # helft van de pixels is 1
    assert mask_area_percentage(mask) == 50.0


def test_mask_area_percentage_respects_threshold():
    mask = np.full((10, 10), 0.4)
    # 0.4 > 0.5 is False, dus 0% boven de standaard drempel
    assert mask_area_percentage(mask, threshold=0.5) == 0.0
    # met een lagere drempel telt het wel mee
    assert mask_area_percentage(mask, threshold=0.3) == 100.0


def test_mask_area_percentage_empty_array():
    mask = np.array([])
    assert mask_area_percentage(mask) == 0.0


@pytest.mark.parametrize("shape,expected_dtype", [((50, 50), float)])
def test_mask_area_percentage_returns_float(shape, expected_dtype):
    mask = np.random.rand(*shape)
    result = mask_area_percentage(mask)
    assert isinstance(result, expected_dtype)
