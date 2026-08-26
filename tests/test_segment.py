import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from gembg.pipeline.segment import cut_out, mask_coverage_ratio

# These tests exercise the real 'u2net' model via rembg. The first run
# on a machine downloads it (~176MB, one-time, cached under
# ~/.rembg/models/u2net); subsequent runs are fast (a few seconds).


def _make_gem_fixture(size=600):
    """A synthetic 'studio photo': light-gray backdrop, a soft drop
    shadow, and a faceted polygon subject -- stands in for a real
    gemstone photo without needing one checked into the repo."""
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


def _make_marquise_fixture(size=520):
    """A synthetic sharp-pointed (marquise-like) gem, to check the
    output stays genuinely pointed rather than getting rounded off."""
    img = Image.new("RGB", (size, size), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    cx = size // 2
    top, bottom = 40, size - 40
    pts = [
        (cx, top),
        (cx + int(size * 0.22), size // 2),
        (cx, bottom),
        (cx - int(size * 0.22), size // 2),
    ]
    draw.polygon(pts, fill=(10, 80, 40))
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        shade = 20 + i * 20
        draw.polygon([a, b, (cx, size // 2)], fill=(shade, 110 + shade, 50 + shade // 2))
    return img.filter(ImageFilter.GaussianBlur(0.4))


def test_cut_out_keeps_sharp_points_sharp():
    # Regression test: u2netp's fixed 320x320 internal resolution rounds
    # off a genuinely sharp point (e.g. a marquise cut's tip) in its own
    # raw prediction -- verified visually on a real marquise emerald
    # photo. cut_out must refine the mask against the full-resolution
    # source (via GrabCut) to recover it, rather than leaving the
    # model's blunted low-res tip as the final output.
    fixture = _make_marquise_fixture()
    cutout = cut_out(fixture)
    alpha = np.array(cutout.split()[-1])

    mask = alpha > 127
    top_row = np.where(mask.any(axis=1))[0].min()

    # 15px below the true drawn apex, a genuinely sharp point should
    # still be narrow. A rounded/blunted tip is much wider here.
    row_width = np.count_nonzero(mask[top_row + 15])
    assert row_width < 25, f"tip looks blunted: {row_width}px wide 15px below the apex"


def test_cut_out_removes_background_and_shadow():
    fixture = _make_gem_fixture()
    cutout = cut_out(fixture)

    assert cutout.mode == "RGBA"
    assert cutout.size == fixture.size

    coverage = mask_coverage_ratio(cutout)
    assert 0.05 < coverage < 0.5, f"unexpected foreground coverage: {coverage}"

    corner_alpha = np.array(cutout.split()[-1])[:20, :20]
    assert corner_alpha.mean() < 10, "corner (background) should be transparent"


def test_cut_out_edges_have_no_background_halo_on_white():
    fixture = _make_gem_fixture()
    cutout = cut_out(fixture)

    # Composite directly (no crop/resize) so pixel positions line up
    # exactly with the alpha array used to pick out edge pixels below.
    white_bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    white_bg.paste(cutout, (0, 0), cutout)
    composited = np.array(white_bg.convert("RGB"))

    alpha = np.array(cutout.split()[-1])
    edge_mask = (alpha > 5) & (alpha < 250)
    assert edge_mask.sum() > 0, "expected some antialiased edge pixels"

    edge_pixels = composited[edge_mask]
    background_gray = np.array([235, 235, 235])
    distance_from_backdrop_gray = np.abs(
        edge_pixels.astype(np.int32) - background_gray
    ).mean()
    assert distance_from_backdrop_gray > 20, (
        "composited edge pixels should not match the original gray "
        "backdrop color (halo)"
    )


def test_cut_out_rgb_matches_source_exactly_even_at_partial_alpha():
    # Regression test: rembg's own RGB output is alpha-premultiplied
    # (raw_rgb == source_rgb * alpha), not straight color. An earlier
    # version used that premultiplied RGB directly, which -- combined
    # with our straight-alpha compositing -- double-multiplied edge
    # pixels by their own alpha, darkening them (visible as a dark rim
    # and dulled facets near edges, not just at the outer silhouette).
    # cut_out must pair the model's alpha with the ORIGINAL source
    # pixels, so RGB matches the source at every alpha level, not just
    # where alpha is 0 or 255.
    fixture = _make_gem_fixture()
    cutout = cut_out(fixture)

    src = np.array(fixture).astype(np.int16)
    out = np.array(cutout)[..., :3].astype(np.int16)
    alpha = np.array(cutout.split()[-1])

    partial = (alpha > 20) & (alpha < 235)  # genuinely partial-alpha pixels
    assert partial.sum() > 0, "expected some partial-alpha edge pixels"

    diff = np.abs(src[partial] - out[partial])
    assert diff.max() <= 2, f"RGB at partial-alpha pixels should match source, got max diff {diff.max()}"
