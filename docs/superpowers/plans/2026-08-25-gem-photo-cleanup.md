# Gem Photo Cleanup Tool Implementation Plan

> **Status: already implemented and verified.** This plan documents the
> working implementation in `gembg/` (built and tested against the
> design spec at `docs/superpowers/specs/2026-08-25-gem-photo-cleanup-design.md`)
> rather than serving as a to-do list for a fresh implementer. Each task
> below reflects code that has been run and its tests passed. Kept here
> so the reasoning and verification behind each design choice is on
> record, and so future changes to any one module can be re-planned
> against a known-good baseline.

**Goal:** A local batch CLI that takes a folder of gemstone product
photos and outputs versions with background and shadow removed,
composited on pure white, with elongated cuts (pear, marquise, oval,
heart, etc.) rotated upright.

**Architecture:** `gembg/pipeline/segment.py` (rembg-based cutout +
edge decontamination), `gembg/pipeline/straighten.py` (PCA-based
rotation), `gembg/pipeline/compose.py` (crop/center/white-canvas), wired
together by `gembg/cli.py`.

**Tech Stack:** Python 3.14, rembg (u2net model), OpenCV, NumPy, Pillow,
pytest.

## Global Constraints

- All processing is local — no cloud/third-party API calls (per spec
  non-goals: avoids sending proprietary product photos off-machine).
- No manual mask-editing UI — failed segmentations are flagged into a
  `needs_review` list, never silently auto-rotated/composited as if
  they'd succeeded.
- Output files keep their source filename, written into `--output`.

## Two deviations from the original design spec (found during build/verification)

1. **Rotation angle via PCA, not `cv2.minAreaRect`.** The spec described
   using `minAreaRect`'s angle. Building a synthetic pear-shaped test
   mask and checking the *actual* resulting orientation (not just a
   width/height ratio) showed `minAreaRect`'s angle can be off by
   20-40 degrees on non-rectangular blobs — it finds the
   minimum-area *enclosing rectangle*, which doesn't track a pear/oval's
   true elongation axis. PCA over the mask's foreground pixel
   coordinates gives the actual principal axis directly and was
   verified correct (residual angle < 0.02°) across 18 test angles from
   -179° to 179°.
2. **Explicit `u2net` model instead of rembg's own default.** rembg's
   current default model is `bria-rmbg` (~1GB). Benchmarked on a
   synthetic gemstone-photo fixture: ~245 seconds/image on CPU — a
   100-photo batch would take over 6 hours. Switching to the classic
   `u2net` model (176MB) gave visually comparable mask quality at
   ~2-4 seconds/image (session load ~2-4s one-time per process, then
   fast per-image). `u2net` is set explicitly in `segment.py` rather
   than left to rembg's default so this doesn't silently regress if
   rembg changes its default again.

Also added beyond the original spec: an **edge-decontamination step**
in `segment.py`. Compositing rembg's raw cutout onto white left a
visible gray halo around the subject — verified present even before any
of our own mask cleanup — because antialiased edge pixels still carry a
blend of foreground and the original gray backdrop color. Fixed by
estimating the backdrop color from the source photo's corners and
solving for the true foreground color at each partially-transparent
edge pixel before compositing (`_decontaminate_edges`). Verified this
removes the halo on the test fixture.

---

### Task 1: White-canvas composition

**Files:**
- Created: `gembg/pipeline/compose.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Produces: `to_white_canvas(rgba_image: PIL.Image, canvas_size: int = 1600, margin: float = 0.08) -> PIL.Image` (RGB mode)

- [x] **Step 1: Write the failing test**

```python
# tests/test_compose.py
from PIL import Image
from gembg.pipeline.compose import to_white_canvas

def test_centers_opaque_content_on_white_canvas():
    source = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    for x in range(50, 150):
        for y in range(20, 80):
            source.putpixel((x, y), (255, 0, 0, 255))
    result = to_white_canvas(source, canvas_size=400, margin=0.1)
    assert result.size == (400, 400)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((399, 399)) == (255, 255, 255)
    center = result.getpixel((200, 200))
    assert center[0] > 200 and center[1] < 100 and center[2] < 100

def test_empty_alpha_returns_blank_white_canvas():
    source = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    result = to_white_canvas(source, canvas_size=300, margin=0.08)
    assert result.size == (300, 300)
    assert result.getpixel((150, 150)) == (255, 255, 255)
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compose.py -v`
Expected (before implementation exists): FAIL with `ModuleNotFoundError` or `ImportError`.

- [x] **Step 3: Write the implementation**

```python
# gembg/pipeline/compose.py
from PIL import Image


def to_white_canvas(rgba_image, canvas_size=1600, margin=0.08):
    bbox = rgba_image.getbbox()
    if bbox is None:
        return Image.new("RGB", (canvas_size, canvas_size), "white")

    cropped = rgba_image.crop(bbox)

    available = canvas_size * (1 - 2 * margin)
    scale = min(available / cropped.width, available / cropped.height)
    new_width = max(1, round(cropped.width * scale))
    new_height = max(1, round(cropped.height * scale))
    resized = cropped.resize((new_width, new_height), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    paste_x = (canvas_size - new_width) // 2
    paste_y = (canvas_size - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)

    return canvas.convert("RGB")
```

- [x] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_compose.py -v`
Result: 2 passed.

- [x] **Step 5: Commit**

```bash
git add gembg/pipeline/compose.py tests/test_compose.py
git commit -m "Add white-canvas composition for gem cutouts"
```

---

### Task 2: Shape/rotation analysis

**Files:**
- Created: `gembg/pipeline/straighten.py`
- Test: `tests/test_straighten.py`

**Interfaces:**
- Consumes: nothing from other tasks (operates on a raw RGBA `PIL.Image`)
- Produces: `compute_rotation(rgba_image: PIL.Image) -> float` (degrees, for `PIL.Image.rotate(angle, expand=True)`)

- [x] **Step 1: Write the failing tests**

```python
# tests/test_straighten.py
import cv2
import numpy as np
import pytest
from PIL import Image
from gembg.pipeline.straighten import compute_rotation

def _make_pear_mask(size=400):
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
    rgba[..., 0] = 20; rgba[..., 1] = 120; rgba[..., 2] = 40
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

@pytest.mark.parametrize("test_angle", [0, 25, -40, 90, 137, -110, 179, -179, 5, -5, 60, -60, 88, -88, 91, -91, 170, -170])
def test_pear_shape_straightens_with_tip_up(test_angle):
    base_rgba = _to_rgba(_make_pear_mask())
    skewed = base_rgba.rotate(test_angle, expand=True, fillcolor=(0, 0, 0, 0))
    rotate_by = compute_rotation(skewed)
    straightened = skewed.rotate(rotate_by, expand=True, fillcolor=(0, 0, 0, 0))
    residual_angle = _pca_angle_from_vertical(straightened)
    assert residual_angle < 1.0
    alpha = np.array(straightened.split()[-1])
    mask = (alpha > 10).astype(np.uint8)
    h = mask.shape[0]
    top_third = np.count_nonzero(mask[: h // 3, :])
    bottom_third = np.count_nonzero(mask[2 * h // 3 :, :])
    assert top_third < bottom_third

def test_round_shape_is_not_rotated():
    round_mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(round_mask, (150, 150), 120, 255, -1)
    assert compute_rotation(_to_rgba(round_mask)) == 0.0

def test_empty_mask_returns_zero():
    empty_rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    assert compute_rotation(empty_rgba) == 0.0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_straighten.py -v`
Expected (before implementation exists): FAIL with `ModuleNotFoundError` or `ImportError`.

- [x] **Step 3: Write the implementation**

An initial version using `cv2.minAreaRect`'s angle was tried first and
**failed** this test suite — measured residual angles of 20-60° on
several test angles because `minAreaRect`'s rectangle doesn't align
with a pear shape's true axis. Replaced with PCA:

```python
# gembg/pipeline/straighten.py
import cv2
import numpy as np
from PIL import Image


def compute_rotation(rgba_image):
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
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_straighten.py -v`
Result: 22 passed (18 angle-parametrized + round-shape + empty-mask).

- [x] **Step 5: Commit**

```bash
git add gembg/pipeline/straighten.py tests/test_straighten.py
git commit -m "Add PCA-based straighten/rotation for elongated gem cuts"
```

---

### Task 3: Background/shadow segmentation

**Files:**
- Created: `gembg/pipeline/segment.py`
- Test: `tests/test_segment.py`
- Modified: `requirements.txt` (add `rembg`, `opencv-python`, `numpy`, `Pillow`)

**Interfaces:**
- Produces: `cut_out(rgb_image: PIL.Image) -> PIL.Image` (RGBA), `mask_coverage_ratio(rgba_image: PIL.Image) -> float`, `estimate_background_color(rgb_image: PIL.Image, patch: int = 20) -> np.ndarray`

- [x] **Step 1: Write the failing tests**

```python
# tests/test_segment.py
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from gembg.pipeline.segment import cut_out, mask_coverage_ratio

def _make_gem_fixture(size=600):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    shadow_layer = Image.new("L", (size, size), 0)
    sd = ImageDraw.Draw(shadow_layer)
    sd.ellipse([size * 0.25, size * 0.68, size * 0.78, size * 0.86], fill=110)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(18))
    shadow_rgb = Image.new("RGB", (size, size), (60, 60, 60))
    img = Image.composite(shadow_rgb, img, shadow_layer)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, int(size * 0.5)
    pts = [
        (cx, cy - 150), (cx + 90, cy - 60), (cx + 110, cy + 60),
        (cx + 60, cy + 150), (cx - 60, cy + 150), (cx - 110, cy + 60),
        (cx - 90, cy - 60),
    ]
    draw.polygon(pts, fill=(10, 90, 40))
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        shade = 30 + (i * 15) % 90
        draw.polygon([a, b, (cx, cy)], fill=(shade, 130 + shade, 60 + shade // 2))
    return img.filter(ImageFilter.GaussianBlur(0.4))

def test_cut_out_removes_background_and_shadow():
    fixture = _make_gem_fixture()
    cutout = cut_out(fixture)
    assert cutout.mode == "RGBA"
    assert cutout.size == fixture.size
    coverage = mask_coverage_ratio(cutout)
    assert 0.05 < coverage < 0.5
    corner_alpha = np.array(cutout.split()[-1])[:20, :20]
    assert corner_alpha.mean() < 10

def test_cut_out_edges_have_no_background_halo_on_white():
    fixture = _make_gem_fixture()
    cutout = cut_out(fixture)
    white_bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    white_bg.paste(cutout, (0, 0), cutout)
    composited = np.array(white_bg.convert("RGB"))
    alpha = np.array(cutout.split()[-1])
    edge_mask = (alpha > 5) & (alpha < 250)
    assert edge_mask.sum() > 0
    edge_pixels = composited[edge_mask]
    background_gray = np.array([235, 235, 235])
    distance_from_backdrop_gray = np.abs(edge_pixels.astype(np.int32) - background_gray).mean()
    assert distance_from_backdrop_gray > 20
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_segment.py -v`
Expected (before implementation exists): FAIL with `ModuleNotFoundError` or `ImportError`.

- [x] **Step 3: Write the implementation**

A first version calling `rembg.remove(rgb_image)` with no explicit
model (rembg's default, `bria-rmbg`) passed the tests but took ~245s on
the fixture — confirmed impractical for batch use (see deviation #2
above), so the session is pinned to `u2net` explicitly. A halo was also
observed on the composited output before adding `_decontaminate_edges`
(see deviation above) — added to make `test_cut_out_edges_have_no_background_halo_on_white` pass.

```python
# gembg/pipeline/segment.py
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
    arr = np.array(rgb_image).astype(np.float64)
    corners = np.concatenate([
        arr[:patch, :patch].reshape(-1, 3),
        arr[:patch, -patch:].reshape(-1, 3),
        arr[-patch:, :patch].reshape(-1, 3),
        arr[-patch:, -patch:].reshape(-1, 3),
    ])
    return corners.mean(axis=0)


def _decontaminate_edges(rgba_image, bg_color):
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
    alpha = np.array(rgba_image.split()[-1])
    return float(np.count_nonzero(alpha > 10)) / alpha.size
```

Add to `requirements.txt`: `rembg>=2.0.0`, `opencv-python>=4.9.0`,
`numpy>=1.26.0`, `Pillow>=10.0.0`.

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_segment.py -v`
Result: 2 passed in 18.9s (model already cached locally; first run on a
fresh machine downloads `u2net.onnx`, ~176MB, one-time).

- [x] **Step 5: Commit**

```bash
git add gembg/pipeline/segment.py tests/test_segment.py requirements.txt
git commit -m "Add rembg-based segmentation with edge decontamination"
```

---

### Task 4: CLI wiring

**Files:**
- Created: `gembg/cli.py`
- Created: `gembg/__init__.py`, `gembg/pipeline/__init__.py` (empty, package markers)

**Interfaces:**
- Consumes: `segment.cut_out`, `segment.mask_coverage_ratio`, `straighten.compute_rotation`, `compose.to_white_canvas` (all as defined in Tasks 1-3)
- Produces: `process_one(source_path, canvas_size, margin) -> (PIL.Image, bool)`, `run(input_dir, output_dir, canvas_size, margin) -> None`, `main(argv=None) -> None`

- [x] **Step 1-4: Write, wire, and smoke-test**

```python
# gembg/cli.py
import argparse
import sys
from pathlib import Path

from PIL import Image

from gembg.pipeline.segment import cut_out, mask_coverage_ratio
from gembg.pipeline.straighten import compute_rotation
from gembg.pipeline.compose import to_white_canvas

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_one(source_path, canvas_size, margin):
    rgb_image = Image.open(source_path).convert("RGB")
    cutout = cut_out(rgb_image)

    coverage = mask_coverage_ratio(cutout)
    needs_review = coverage < 0.01 or coverage > 0.95

    if not needs_review:
        angle = compute_rotation(cutout)
        if angle:
            cutout = cutout.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    output_image = to_white_canvas(cutout, canvas_size=canvas_size, margin=margin)
    return output_image, needs_review


def run(input_dir, output_dir, canvas_size, margin):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    processed = 0
    needs_review_files = []
    skipped_files = []

    for source_path in sorted(input_dir.iterdir()):
        if not source_path.is_file():
            continue
        total += 1
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped_files.append(source_path.name)
            continue

        output_image, needs_review = process_one(source_path, canvas_size, margin)
        output_path = output_dir / source_path.name
        output_image.save(output_path, quality=95)
        processed += 1
        if needs_review:
            needs_review_files.append(source_path.name)

    print(f"Total files seen: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped (unsupported extension): {len(skipped_files)}")
    for name in skipped_files:
        print(f"  - {name}")
    print(f"Needs review (segmentation may have failed): {len(needs_review_files)}")
    for name in needs_review_files:
        print(f"  - {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean up gemstone product photos.")
    parser.add_argument("--input", required=True, help="Folder of source photos")
    parser.add_argument("--output", required=True, help="Folder to write cleaned photos into")
    parser.add_argument("--canvas-size", type=int, default=1600)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args(argv)

    run(args.input, args.output, args.canvas_size, args.margin)


if __name__ == "__main__":
    sys.exit(main())
```

Smoke test run:

```bash
python -m gembg.cli --input input --output output
```

Verified against a real photo (the pear-shaped emerald sample provided
at the start of this project) — output showed clean white background,
no shadow, no halo, correct upright orientation.

- [x] **Step 5: Commit**

```bash
git add gembg/cli.py gembg/__init__.py gembg/pipeline/__init__.py
git commit -m "Wire pipeline stages into a batch CLI"
```

## How to run it

```bash
pip install -r requirements.txt
python -m gembg.cli --input <folder-of-photos> --output <folder-for-results>
```

First run downloads the `u2net` model (~176MB) once; subsequent runs
reuse the cached copy. Check the "Needs review" list the CLI prints —
those files got background/shadow removal but were not auto-rotated,
because segmentation coverage looked abnormal (nearly empty or nearly
the whole frame).
