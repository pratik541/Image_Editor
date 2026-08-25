# Gem Photo Cleanup Tool — Design

## Purpose

Poddar Diamonds product photos of loose gemstones are shot on a light-gray
studio backdrop with a visible drop shadow. We need a repeatable, local
batch tool that takes a folder of these photos and produces catalog-ready
versions: background and shadow removed, composited onto a pure white
canvas, and straightened so elongated cuts (pear, marquise, oval, heart,
emerald-cut, etc.) sit upright with a consistent orientation.

Round/near-square cuts (round brilliant, cushion, princess) have no
meaningful "upright" orientation and are only background/shadow-cleaned,
not rotated.

## Non-goals

- No 3D reconstruction or synthetic "front-facing" view generation — this
  operates on the existing 2D photo only.
- No cloud/third-party API calls — all processing is local (avoids sending
  unreleased/proprietary product photos to a third party).
- No manual mask-editing UI in v1 — failures are flagged for manual review
  instead of guessed at.

## Approach

AI segmentation via `rembg` (local ONNX model, downloaded once on first
run) produces an alpha mask that separates the gem from both the
background and its cast shadow in one pass. OpenCV is then used on that
mask to find the gem's contour and orientation. This was chosen over pure
classical CV (color-distance thresholding / GrabCut) because the sample
photos have strong internal facet reflections that reliably break
threshold-based masking (holes/bites in the cutout). A cloud background-removal
API was ruled out per the non-goals above.

## Architecture

```
gembg/
  cli.py                 # entry point, folder walk, logging/summary
  pipeline/
    segment.py           # rembg wrapper -> RGBA cutout
    straighten.py         # contour analysis, shape classification, rotation
    compose.py            # crop, center, composite onto white canvas
```

Each `pipeline/*.py` module is a pure function over image data (no I/O),
so they can be tested independently:

- `segment.cut_out(rgb_image) -> rgba_image`
- `straighten.compute_rotation(rgba_image) -> angle_degrees` (0 if shape is
  round/near-square or if orientation can't be determined confidently)
- `compose.to_white_canvas(rgba_image, canvas_size, margin) -> rgb_image`

`cli.py` composes these three per file and handles all filesystem/logging
concerns.

## Pipeline steps (per image)

1. **Load** the source image (JPEG/PNG).
2. **Segment**: run `rembg` to get an RGBA cutout. This removes both
   background and drop shadow, since the model segments the subject
   rather than keying on background color.
3. **Clean mask**: morphological close to fill small holes caused by
   bright facet reflections; slight Gaussian feather on the alpha edge so
   the final cutout doesn't look hard-edged/pasted.
4. **Analyze shape & straighten**:
   - Find the largest contour in the alpha mask.
   - Compute `cv2.minAreaRect` → bounding box aspect ratio and angle.
   - If aspect ratio is within ~15% of 1:1, treat as round/near-square →
     skip rotation (angle = 0).
   - Otherwise, rotate so the long axis is vertical. To decide which end
     is "up": compare the mask's centroid position against the midpoint
     of the long axis — the narrower end (smaller mask area near that tip)
     is rotated to the top. This handles pear/heart shapes where one end
     is visibly narrower than the other; for symmetric elongated shapes
     (oval, marquise, emerald-cut) either orientation is visually
     equivalent, so no flip decision is needed.
5. **Compose**: crop to the gem's bounding box plus an 8% margin, resize
   to fit within a square canvas (default 1600×1600, configurable), paste
   centered onto a solid white background, flatten alpha, save as JPEG
   (quality 95) using the original filename into the output folder.

## CLI

```
python cli.py --input <folder> --output <folder> [--canvas-size 1600] [--margin 0.08]
```

- `--input`: folder of source photos (JPEG/PNG; other files are skipped
  with a warning).
- `--output`: folder to write cleaned photos into (created if missing).
  Output files keep the same filename as their source.
- `--canvas-size`: side length in pixels of the square white output canvas.
- `--margin`: fraction of canvas size left as empty margin around the gem.

## Error handling

- If `rembg`'s mask is empty (nothing detected) or covers ~the whole
  frame (segmentation failed to separate subject from background), the
  tool still produces a background/white-composited output using
  whatever mask it got, but **skips the rotation step** and adds the
  filename to a `needs_review` list.
- Unsupported file extensions are skipped with a logged warning, not an
  error that halts the batch.
- At the end of a run, the CLI prints a summary: total files, processed
  successfully, count needing review (with filenames), count skipped.

## Post-implementation notes

Three refinements were made during implementation and verification (see
`docs/superpowers/plans/2026-08-25-gem-photo-cleanup.md` for the
evidence behind each):

- Rotation angle is computed via PCA over the alpha mask's foreground
  pixels, not `cv2.minAreaRect`. Testing on a synthetic pear-shaped mask
  showed `minAreaRect`'s angle can be off by tens of degrees for
  non-rectangular blobs.
- `segment.py` pins rembg to the `u2net` model explicitly. rembg's own
  default model (`bria-rmbg`) measured ~245s/image on CPU in testing —
  impractical for batch use — versus ~2-4s/image for `u2net` with
  comparable mask quality on our test photos.
- An edge-decontamination step was added to `segment.py`
  (`_decontaminate_edges`), beyond what was originally scoped. Without
  it, compositing the cutout onto white left a visible gray halo around
  the subject from antialiased edge pixels still carrying backdrop
  color.

## Testing

- Unit-level: feed synthetic/sample RGBA masks (round mask, pear-shaped
  mask oriented at a known angle) into `straighten.compute_rotation` and
  assert expected angle/no-rotation decisions.
- End-to-end smoke test: run the CLI on the single sample pear-shaped
  emerald photo provided, inspect the output visually before running on
  a full batch.
