import cv2
import numpy as np
import pytest
from PIL import Image

from gembg.pipeline.straighten import compute_rotation


def _make_pear_mask(size=400):
    """Wide circle (bottom) + narrow circle (top tip) + connecting
    triangle: an upright pear silhouette, before any test rotation."""
    mask = np.zeros((size, size), dtype=np.uint8)
    wide_center = (size // 2, int(size * 0.68))
    wide_r = int(size * 0.28)
    narrow_center = (size // 2, int(size * 0.22))
    narrow_r = int(size * 0.10)
    cv2.circle(mask, wide_center, wide_r, 255, -1)
    cv2.circle(mask, narrow_center, narrow_r, 255, -1)
    pts = np.array([
        [wide_center[0] - wide_r + 10, wide_center[1] - 10],
        [wide_center[0] + wide_r - 10, wide_center[1] - 10],
        [narrow_center[0] + narrow_r, narrow_center[1] + 5],
        [narrow_center[0] - narrow_r, narrow_center[1] + 5],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _to_rgba(mask):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 20
    rgba[..., 1] = 120
    rgba[..., 2] = 40
    rgba[..., 3] = mask
    return Image.fromarray(rgba, mode="RGBA")


def _pca_angle_from_vertical(rgba_image):
    alpha = np.array(rgba_image.split()[-1])
    mask = (alpha > 10).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float64)
    _, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
    vx, vy = eigenvectors[0]
    angle = np.degrees(np.arctan2(vx, vy))
    return min(abs(angle), abs(180 - abs(angle)))


@pytest.mark.parametrize(
    "test_angle",
    [0, 25, -40, 90, 137, -110, 179, -179, 5, -5, 60, -60, 88, -88, 91, -91, 170, -170],
)
def test_pear_shape_straightens_with_tip_up(test_angle):
    base_rgba = _to_rgba(_make_pear_mask())
    skewed = base_rgba.rotate(test_angle, expand=True, fillcolor=(0, 0, 0, 0))

    rotate_by = compute_rotation(skewed)
    straightened = skewed.rotate(rotate_by, expand=True, fillcolor=(0, 0, 0, 0))

    residual_angle = _pca_angle_from_vertical(straightened)
    assert residual_angle < 1.0, (
        f"residual angle too large for test_angle={test_angle}: {residual_angle}"
    )

    alpha = np.array(straightened.split()[-1])
    mask = (alpha > 10).astype(np.uint8)
    h = mask.shape[0]
    top_third = np.count_nonzero(mask[: h // 3, :])
    bottom_third = np.count_nonzero(mask[2 * h // 3 :, :])
    assert top_third < bottom_third, f"tip not pointing up for test_angle={test_angle}"


def test_round_shape_is_not_rotated():
    round_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(round_mask, (150, 150), 120, 255, -1)
    round_rgba = _to_rgba(round_mask)
    assert compute_rotation(round_rgba) == 0.0


def test_empty_mask_returns_zero():
    empty_rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert compute_rotation(empty_rgba) == 0.0
