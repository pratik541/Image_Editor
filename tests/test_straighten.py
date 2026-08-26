import cv2
import numpy as np
import pytest
from PIL import Image

from gembg.pipeline.straighten import compute_rotation, rotate_rgba


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


def test_rotate_rgba_angle_zero_is_a_no_op():
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    rgba[10:40, 10:40] = (30, 200, 90, 255)
    image = Image.fromarray(rgba, mode="RGBA")

    result = rotate_rgba(image, 0.0)

    assert result is image  # explicit fast path, not just pixel-equal


def test_rotate_rgba_preserves_fully_opaque_interior_color():
    # The premultiply/rotate/un-premultiply round trip must be a no-op
    # for pixels that stay fully opaque and are far from any edge --
    # only partial-alpha edge pixels should be affected.
    size = 200
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= (size * 0.4) ** 2
    rgba[disk] = (30, 180, 90, 255)
    image = Image.fromarray(rgba, mode="RGBA")

    rotated = rotate_rgba(image, 40.0)
    arr = np.array(rotated)

    # Sample well inside the rotated disk, away from any edge.
    cy, cx = rotated.size[1] // 2, rotated.size[0] // 2
    center_pixel = arr[cy, cx]
    assert center_pixel[3] == 255
    assert tuple(center_pixel[:3]) == (30, 180, 90)


def test_rotate_rgba_matches_manual_premultiply_unpremultiply():
    # Pins the exact algorithm (not just "looks reasonable"): running
    # the same premultiply -> PIL bicubic rotate -> un-premultiply steps
    # independently must reproduce rotate_rgba's output exactly.
    size = 80
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, size=(size, size, 4), dtype=np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")

    arr = np.array(image).astype(np.float64)
    alpha = arr[..., 3:4] / 255.0
    premultiplied = arr.copy()
    premultiplied[..., :3] = arr[..., :3] * alpha
    pm_image = Image.fromarray(premultiplied.astype(np.uint8), mode="RGBA")
    expected_rotated = pm_image.rotate(
        22.0, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
    )
    expected_arr = np.array(expected_rotated).astype(np.float64)
    expected_alpha = expected_arr[..., 3:4] / 255.0
    safe_alpha = np.where(expected_alpha > (1 / 255), expected_alpha, 1.0)
    expected_arr[..., :3] = np.clip(expected_arr[..., :3] / safe_alpha, 0, 255)
    expected = expected_arr.astype(np.uint8)

    actual = np.array(rotate_rgba(image, 22.0))

    assert np.array_equal(actual, expected)
