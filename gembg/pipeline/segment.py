import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = new_session("u2netp")
    return _SESSION


def cut_out(rgb_image):
    """Run background/shadow removal and return a cleaned RGBA cutout:
    small holes in the alpha mask (from bright facet reflections) are
    closed, and the alpha edge is softly feathered.

    rembg's returned RGB is alpha-premultiplied (verified: raw_rgb ==
    source_rgb * alpha to within uint8 rounding), not the straight
    color our compositing (and an earlier decontamination step here)
    assumed. Treating it as straight alpha double-multiplied edge
    pixels by their own alpha, darkening them -- visible as a dark rim
    and dulled facet color near edges. Fixed by pairing the model's
    alpha mask with the ORIGINAL source pixels instead of rembg's RGB,
    which sidesteps the premultiplication entirely.

    Uses the 'u2netp' model explicitly rather than rembg's own default
    ('bria-rmbg', a ~1GB model that measured ~245s/image on CPU -- see
    git history). 'u2netp' (~4.5MB) was chosen over the mid-size
    'u2net' (~176MB) specifically for memory/bandwidth-constrained
    hosting (e.g. Streamlit Community Cloud's free tier): comparable
    mask quality and coverage on our test photos, ~2x faster inference,
    and a download small enough to not risk stalling on a slow or
    constrained network path."""
    result = remove(rgb_image, session=_get_session())
    alpha = np.array(result.split()[-1])

    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    feathered = cv2.GaussianBlur(closed, (5, 5), 0)

    r, g, b = rgb_image.split()
    return Image.merge("RGBA", (r, g, b, Image.fromarray(feathered)))


def mask_coverage_ratio(rgba_image):
    """Fraction of pixels considered foreground; used by the CLI to
    detect a failed segmentation (empty mask or ~whole-frame mask)."""
    alpha = np.array(rgba_image.split()[-1])
    return float(np.count_nonzero(alpha > 10)) / alpha.size
