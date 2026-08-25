import numpy as np
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


def test_does_not_upscale_content_smaller_than_available_area():
    # A small opaque region on a much larger canvas should keep its
    # native pixel size (more white margin) instead of being blown up
    # via interpolation, which would blur it. Regression test: an
    # earlier version scaled up to *fill* the available area regardless
    # of the source's native resolution.
    source = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    for x in range(10, 50):
        for y in range(5, 35):
            source.putpixel((x, y), (255, 0, 0, 255))

    result = to_white_canvas(source, canvas_size=1600, margin=0.08)

    non_white = np.any(np.array(result) != 255, axis=-1)
    ys, xs = np.nonzero(non_white)
    assert xs.max() - xs.min() + 1 == 40  # native bbox width, not upscaled
    assert ys.max() - ys.min() + 1 == 30  # native bbox height, not upscaled
