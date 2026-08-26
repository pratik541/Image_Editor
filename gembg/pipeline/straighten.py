import cv2
import numpy as np
from PIL import Image


def compute_rotation(rgba_image):
    """Angle in degrees to pass to PIL's Image.rotate(angle, expand=True)
    so the shape's long axis becomes vertical with its narrower end up.

    Uses PCA over the alpha mask's foreground pixels rather than
    cv2.minAreaRect, because minAreaRect's minimum-enclosing-rectangle
    angle does not reliably track a blob's true elongation axis for
    non-rectangular shapes like a pear or oval gem (verified: it can be
    off by tens of degrees on a pear-shaped test mask). PCA's principal
    axis matches the shape's actual long axis directly.

    Returns 0.0 for round/near-square shapes (major/minor eigenvalue
    ratio below 1.4, empirically close to the spec's "aspect ratio
    within ~15% of 1:1") or if fewer than 2 foreground pixels are found.
    """
    alpha = np.array(rgba_image.split()[-1])
    mask = (alpha > 10).astype(np.uint8) * 255

    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0

    points = np.column_stack([xs, ys]).astype(np.float64)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)

    eigenvalues = eigenvalues.flatten()
    major_eigenvalue, minor_eigenvalue = eigenvalues[0], eigenvalues[1]
    if minor_eigenvalue <= 0 or (major_eigenvalue / minor_eigenvalue) < 1.4:
        return 0.0

    vx, vy = eigenvectors[0]
    rotate_by = -float(np.degrees(np.arctan2(vx, vy)))

    rotated_mask = np.array(Image.fromarray(mask).rotate(rotate_by, expand=True))
    top_area, bottom_area = _half_areas(rotated_mask)
    if top_area > bottom_area:
        rotate_by += 180.0

    return rotate_by


def _half_areas(mask):
    h = mask.shape[0]
    top = mask[: h // 2, :]
    bottom = mask[h // 2 :, :]
    return int(np.count_nonzero(top)), int(np.count_nonzero(bottom))


def rotate_rgba(rgba_image, angle):
    """Rotate an RGBA image by angle degrees (bicubic, expanding the
    canvas), without color fringing at the transparent edge.

    Naively rotating straight (non-premultiplied) RGBA with bicubic
    interpolation blends full-strength color from adjacent pixels into
    semi-transparent edge pixels independent of alpha. Verified on a
    real photo: a bright facet reflection sitting right next to dark
    facets, at the gem's silhouette edge, produced a visible wavy
    discoloration after rotation that was confirmed absent before
    rotating (same cutout, composited onto white with no rotation was
    clean). Premultiplying by alpha before rotating, and un-premultiplying
    after, is the standard fix: it makes near-transparent pixels
    contribute close to black to the interpolation instead of their
    full unmasked color, eliminating the fringing -- verified fixed on
    the same photo."""
    if not angle:
        return rgba_image

    arr = np.array(rgba_image).astype(np.float64)
    alpha = arr[..., 3:4] / 255.0
    premultiplied = arr.copy()
    premultiplied[..., :3] = arr[..., :3] * alpha
    pm_image = Image.fromarray(premultiplied.astype(np.uint8), mode="RGBA")

    rotated = pm_image.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
    )

    rotated_arr = np.array(rotated).astype(np.float64)
    rotated_alpha = rotated_arr[..., 3:4] / 255.0
    safe_alpha = np.where(rotated_alpha > (1 / 255), rotated_alpha, 1.0)
    unpremultiplied_rgb = np.clip(rotated_arr[..., :3] / safe_alpha, 0, 255)
    rotated_arr[..., :3] = np.where(
        rotated_alpha > (1 / 255), unpremultiplied_rgb, rotated_arr[..., :3]
    )

    return Image.fromarray(rotated_arr.astype(np.uint8), mode="RGBA")
