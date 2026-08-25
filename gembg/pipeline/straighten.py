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
