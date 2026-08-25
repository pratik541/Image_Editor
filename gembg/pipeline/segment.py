import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = new_session("u2net")
    return _SESSION


def estimate_background_color(rgb_image, patch=20):
    """Average color of the four corners, used as the studio backdrop
    color estimate for edge decontamination."""
    arr = np.array(rgb_image).astype(np.float64)
    corners = np.concatenate([
        arr[:patch, :patch].reshape(-1, 3),
        arr[:patch, -patch:].reshape(-1, 3),
        arr[-patch:, :patch].reshape(-1, 3),
        arr[-patch:, -patch:].reshape(-1, 3),
    ])
    return corners.mean(axis=0)


def _decontaminate_edges(rgba_image, bg_color):
    """Remove residual backdrop-color spill from partially-transparent
    edge pixels, so compositing onto white doesn't leave a gray/dark
    halo around the cutout. Verified necessary: rembg's raw output
    still shows a visible gray ring around the subject when pasted
    onto white without this step."""
    arr = np.array(rgba_image).astype(np.float64)
    alpha = arr[..., 3] / 255.0
    rgb = arr[..., :3]
    bg = np.array(bg_color, dtype=np.float64)

    safe_alpha = np.where(alpha > 0.02, alpha, 1.0)[..., None]
    foreground = (rgb - (1 - alpha[..., None]) * bg) / safe_alpha
    foreground = np.clip(foreground, 0, 255)

    edge_mask = (alpha > 0.02)[..., None]
    arr[..., :3] = np.where(edge_mask, foreground, rgb)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def cut_out(rgb_image):
    """Run background/shadow removal and return a cleaned RGBA cutout:
    edge pixels are decontaminated (no backdrop-color halo when later
    composited onto white), small holes in the alpha mask (from bright
    facet reflections) are closed, and the alpha edge is softly
    feathered.

    Uses the 'u2net' model explicitly rather than rembg's own default
    ('bria-rmbg'), which measured ~245s/image on CPU in testing vs.
    ~2-4s/image for u2net with comparable mask quality on our sample
    photos -- bria-rmbg is a ~1GB model that's impractical for batch use
    without a GPU."""
    bg_color = estimate_background_color(rgb_image)
    result = remove(rgb_image, session=_get_session())
    if result.mode != "RGBA":
        result = result.convert("RGBA")

    result = _decontaminate_edges(result, bg_color)

    r, g, b, a = result.split()
    alpha = np.array(a)

    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    feathered = cv2.GaussianBlur(closed, (5, 5), 0)

    return Image.merge("RGBA", (r, g, b, Image.fromarray(feathered)))


def mask_coverage_ratio(rgba_image):
    """Fraction of pixels considered foreground; used by the CLI to
    detect a failed segmentation (empty mask or ~whole-frame mask)."""
    alpha = np.array(rgba_image.split()[-1])
    return float(np.count_nonzero(alpha > 10)) / alpha.size
