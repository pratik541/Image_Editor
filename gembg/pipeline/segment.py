from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import pooch
from PIL import Image

# Two models, deliberately kept as a user-facing choice rather than one
# default: BiRefNet is far sharper (runs at 1024x1024 internally) but
# measured ~75-85s/image on CPU; u2netp is ~1s/image but visibly
# blunter/softer at edges -- confirmed side-by-side on the same real
# photo, even with every other fix in this file already applied. There
# is no configuration that closes this gap; it's a genuine trade-off.
MODELS = {
    "fast": {
        "url": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "checksum": "md5:8e83ca70e441ab06c318d82300c84806",
        "filename": "u2netp.onnx",
        "input_size": (320, 320),
        "use_sigmoid": False,
    },
    "quality": {
        "url": (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"
        ),
        "checksum": "md5:4fab47adc4ff364be1713e97b7e66334",
        "filename": "birefnet-general-lite.onnx",
        "input_size": (1024, 1024),
        "use_sigmoid": True,
    },
}
DEFAULT_MODEL = "quality"

_MODEL_CACHE_DIR = Path.home() / ".gembg" / "models"
_MODEL_MEAN = (0.485, 0.456, 0.406)
_MODEL_STD = (0.229, 0.224, 0.225)

_SESSIONS = {}


def _get_session(model):
    """Load the given model's ONNX file directly via onnxruntime,
    bypassing the rembg *package* entirely (we only ever used its
    session-runner, never its alpha-matting features).

    rembg's top-level import pulls in pymatting, which pulls in numba --
    a native JIT compiler that hung indefinitely (unrecoverable even with
    a Python-level timeout, since a native/GIL-holding hang can't be
    preempted from Python) on one hosting platform. onnxruntime + numpy +
    PIL + pooch, used directly, provide the exact same inference this app
    needs without that dependency chain. Preprocessing/postprocessing
    below is copied from rembg's own session classes, verified by
    reading their source (rembg/sessions/base.py, u2netp.py,
    birefnet_general.py)."""
    if model not in _SESSIONS:
        config = MODELS[model]
        model_path = pooch.retrieve(
            config["url"],
            config["checksum"],
            fname=config["filename"],
            path=_MODEL_CACHE_DIR,
            progressbar=False,
        )
        # Disable the memory arena and pre-computed memory pattern.
        # Measured with peak-RSS polling during a real "quality" model
        # inference: default settings peaked at ~5.0GB, this config at
        # ~3.9GB (~22% less) -- and was faster too, not a trade-off.
        # Still nowhere near low enough for a ~1GB-RAM free-tier host
        # (that's an architectural property of a 1024x1024 Swin
        # Transformer, not something a session option can fix), but a
        # real, verified improvement worth keeping regardless.
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        _SESSIONS[model] = onnxruntime.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[model]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _predict_alpha_mask(session, rgb_image, config):
    """Run the model on rgb_image and return a single-channel 'L' mask at
    the image's own resolution."""
    resized = rgb_image.convert("RGB").resize(
        config["input_size"], Image.Resampling.LANCZOS
    )

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
    if config["use_sigmoid"]:
        prediction = _sigmoid(prediction)
    lo, hi = prediction.min(), prediction.max()
    prediction = (prediction - lo) / (hi - lo)
    prediction = np.squeeze(prediction)

    mask = Image.fromarray((prediction.clip(0, 1) * 255).astype("uint8"), mode="L")
    return mask.resize(rgb_image.size, Image.Resampling.LANCZOS)


def _refine_with_grabcut(alpha, rgb_image):
    """Sharpen the model's mask against the source image's actual
    full-resolution color data.

    Kept as a cheap (~0.6s) extra safety net even for the higher-
    resolution model (1024x1024): still a hardcoded input size smaller
    than most source photos, so a genuinely sharp point can still lose
    a little precision in the model's own prediction. GrabCut, seeded
    with this mask as a trimap, re-derives the boundary from the real
    image at full resolution. Verified on two test photos: recovers a
    visibly sharper point, coverage barely shifts (<0.1%), and no new
    holes appear elsewhere in the mask.

    Falls back to the unrefined mask if GrabCut fails on a degenerate
    trimap (e.g. an almost entirely empty or full mask), rather than
    letting the whole pipeline crash on an edge case."""
    trimap = np.full(alpha.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    trimap[alpha > 200] = cv2.GC_FGD
    trimap[alpha < 20] = cv2.GC_BGD
    trimap[(alpha >= 20) & (alpha <= 200)] = cv2.GC_PR_FGD

    if not (trimap == cv2.GC_FGD).any() or not (trimap == cv2.GC_BGD).any():
        return alpha

    bgr = cv2.cvtColor(np.array(rgb_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, trimap, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return alpha

    is_foreground = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
    return np.where(is_foreground, 255, 0).astype(np.uint8)


def cut_out(rgb_image, model=DEFAULT_MODEL):
    """Run background/shadow removal and return a cleaned RGBA cutout:
    small holes in the alpha mask (from bright facet reflections) are
    closed, the boundary is sharpened against the full-resolution
    source image, and the alpha edge is softly feathered.

    model: "quality" (default, BiRefNet-general-lite, ~75-85s/image on
    CPU, sharp edges) or "fast" (u2netp, ~1s/image, visibly softer
    edges) -- see MODELS above.

    Pairs the model's alpha mask directly with the ORIGINAL source
    pixels (not a matted/recolored RGB), which is both simpler and
    avoids a premultiplied-alpha bug rembg's own RGB output had (see
    git history: rembg's raw_rgb == source_rgb * alpha, not straight
    color -- verified darkened/dulled edges when treated as straight
    alpha)."""
    config = MODELS[model]
    alpha = np.array(_predict_alpha_mask(_get_session(model), rgb_image, config))

    # 21x21, not 7x7: verified on a real photo that bright reflection
    # facets whose color sits close to the gray backdrop (found two, at
    # the crown's top-center and bottom-center) can locally dip the
    # model's confidence right at the boundary, leaving small inward
    # notches a 7x7 close doesn't bridge. 21x21 closes both cleanly
    # (coverage unchanged to 4 decimal places -- confirms it's a
    # targeted fix, not a broad shape change) without rounding a
    # genuinely sharp point any more than 7x7 does (compared
    # side-by-side on the same photo's marquise tip).
    kernel = np.ones((21, 21), np.uint8)
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    refined = _refine_with_grabcut(closed, rgb_image)
    feathered = cv2.GaussianBlur(refined, (5, 5), 0)

    r, g, b = rgb_image.split()
    return Image.merge("RGBA", (r, g, b, Image.fromarray(feathered)))


def mask_coverage_ratio(rgba_image):
    """Fraction of pixels considered foreground; used by the CLI to
    detect a failed segmentation (empty mask or ~whole-frame mask)."""
    alpha = np.array(rgba_image.split()[-1])
    return float(np.count_nonzero(alpha > 10)) / alpha.size
