from PIL import Image


def to_white_canvas(rgba_image, canvas_size=1600, margin=0.08):
    """Crop an RGBA cutout to its alpha bounding box, center it on a
    square white canvas with the given fractional margin, and return
    a flattened RGB image."""
    bbox = rgba_image.getbbox()
    if bbox is None:
        return Image.new("RGB", (canvas_size, canvas_size), "white")

    cropped = rgba_image.crop(bbox)

    # Never resample a gem that already fits on the canvas at its native
    # resolution -- any resize (even shrinking) resamples pixels, which
    # softens fine facet detail. `margin` is only a target when we must
    # shrink anyway (crop bigger than the canvas itself); it must not
    # force a shrink of a gem that would otherwise fit as-is. Confirmed
    # by measurement: even a modest shrink-to-fit-margin on an
    # already-fitting crop is enough to visibly soften facet edges.
    if cropped.width <= canvas_size and cropped.height <= canvas_size:
        scale = 1.0
    else:
        target = canvas_size * (1 - 2 * margin)
        scale = min(target / cropped.width, target / cropped.height)

    new_width = max(1, round(cropped.width * scale))
    new_height = max(1, round(cropped.height * scale))
    resized = cropped if scale == 1.0 else cropped.resize((new_width, new_height), Image.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    paste_x = (canvas_size - new_width) // 2
    paste_y = (canvas_size - new_height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)

    return canvas.convert("RGB")
