from PIL import Image


def to_white_canvas(rgba_image, canvas_size=1600, margin=0.08):
    """Crop an RGBA cutout to its alpha bounding box, center it on a
    square white canvas with the given fractional margin, and return
    a flattened RGB image."""
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
