from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import pooch
from PIL import Image

_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
_MODEL_CHECKSUM = "md5:8e83ca70e441ab06c318d82300c84806"
_MODEL_CACHE_DIR = Path.home() / ".gembg" / "models"
_MODEL_INPUT_SIZE = (320, 320)
_MODEL_MEAN = (0.485, 0.456, 0.406)
_MODEL_STD = (0.229, 0.224, 0.225)

_SESSION = None


def _get_session():
    """Load the u2netp ONNX model directly via onnxruntime, bypassing the
    rembg *package* entirely (we only ever used its session-runner, never
    its alpha-matting features).

    rembg's top-level import pulls in pymatting, which pulls in numba --
    a native JIT compiler that hung indefinitely (unrecoverable even with
    a Python-level timeout, since a native/GIL-holding hang can't be
    preempted from Python) on one hosting platform. onnxruntime + numpy +
    PIL + pooch, used directly, provide the exact same inference this app
    needs without that dependency chain. Preprocessing/postprocessing
    below is copied from rembg's own U2netpSession, verified by reading
    its source (rembg/sessions/base.py, u2netp.py)."""
    global _SESSION
    if _SESSION is None:
        model_path = pooch.retrieve(
            _MODEL_URL,
            _MODEL_CHECKSUM,
            fname="u2netp.onnx",
            path=_MODEL_CACHE_DIR,
            progressbar=False,
        )
        _SESSION = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    return _SESSION


def _predict_alpha_mask(session, rgb_image):
    """Run u2netp on rgb_image and return a single-channel 'L' mask at the
    image's own resolution."""
    resized = rgb_image.convert("RGB").resize(_MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)

    array = np.array(resized) / max(np.array(resized).max(), 1e-6)
    normalized = np.zeros_like(array, dtype=np.float32)
    for channel in range(3):
        normalized[:, :, channel] = (
            array[:, :, channel] - _MODEL_MEAN[channel]
        ) / _MODEL_STD[channel]
    tensor = np.expand_dims(normalized.transpose((2, 0, 1)), 0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    prediction = outputs[0][:, 0, :, :]
    lo, hi = prediction.min(), prediction.max()
    prediction = (prediction - lo) / (hi - lo)
    prediction = np.squeeze(prediction)

    mask = Image.fromarray((prediction.clip(0, 1) * 255).astype("uint8"), mode="L")
    return mask.resize(rgb_image.size, Image.Resampling.LANCZOS)


def cut_out(rgb_image):
    """Run background/shadow removal and return a cleaned RGBA cutout:
    small holes in the alpha mask (from bright facet reflections) are
    closed, and the alpha edge is softly feathered.

    Pairs the model's alpha mask directly with the ORIGINAL source
    pixels (not a matted/recolored RGB), which is both simpler and
    avoids a premultiplied-alpha bug rembg's own RGB output had (see
    git history: rembg's raw_rgb == source_rgb * alpha, not straight
    color -- verified darkened/dulled edges when treated as straight
    alpha)."""
    alpha = np.array(_predict_alpha_mask(_get_session(), rgb_image))

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
