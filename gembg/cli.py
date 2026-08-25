import argparse
import sys
from pathlib import Path

from PIL import Image

from gembg.pipeline.segment import cut_out, mask_coverage_ratio
from gembg.pipeline.straighten import compute_rotation
from gembg.pipeline.compose import to_white_canvas

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_one(source_path, canvas_size, margin):
    """Returns (output_image, needs_review: bool)."""
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
