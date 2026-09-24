import json
from pathlib import Path

import numpy as np
import pytest

from landmarks import augment, normalize_landmarks

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def reference():
    return json.loads((FIXTURES / "normalization.json").read_text())


def test_matches_reference_values(reference):
    # The same values are checked by the Android and web tests.
    result = normalize_landmarks(reference["landmarks"], reference["width"], reference["height"])
    np.testing.assert_allclose(result.flatten(), reference["expected"], atol=1e-5)


def test_wrist_is_origin_and_hand_has_unit_size(reference):
    result = normalize_landmarks(reference["landmarks"], reference["width"], reference["height"])
    np.testing.assert_allclose(result[0], 0, atol=1e-7)
    assert np.linalg.norm(result[:, :2], axis=1).max() == pytest.approx(1.0)


def test_ignores_hand_position_and_size(reference):
    points = np.array(reference["landmarks"])
    moved = points * 0.5 + np.array([0.1, 0.2, 0.0])
    np.testing.assert_allclose(
        normalize_landmarks(points, 480, 640),
        normalize_landmarks(moved, 480, 640),
        atol=1e-5,
    )


def test_accounts_for_image_aspect_ratio():
    # The same physical hand in a portrait and a landscape frame of equal height.
    points = np.array([[0.5, 0.5, 0.0], [0.6, 0.5, 0.0], [0.5, 0.4, 0.0]] + [[0.5, 0.5, 0.0]] * 18)
    landscape = points.copy()
    landscape[:, 0] = 0.5 + (points[:, 0] - 0.5) * 480 / 640
    np.testing.assert_allclose(
        normalize_landmarks(points, 480, 640),
        normalize_landmarks(landscape, 640, 640),
        atol=1e-5,
    )


def test_augment_keeps_originals_and_adds_copies(reference):
    points = normalize_landmarks(reference["landmarks"], 480, 640)[None].repeat(4, axis=0)
    result = augment(points, copies=3, rng=np.random.default_rng(0))
    assert result.shape == (16, 21, 3)
    np.testing.assert_array_equal(result[:4], points)
    assert not np.allclose(result[4:], np.tile(points, (3, 1, 1)))
