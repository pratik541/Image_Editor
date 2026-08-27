from unittest.mock import patch

from PIL import Image

from gembg.cli import process_one, resolve_canvas_size


def test_none_matches_larger_side_of_source_image():
    assert resolve_canvas_size(None, (518, 518)) == 518
    assert resolve_canvas_size(None, (600, 400)) == 600
    assert resolve_canvas_size(None, (400, 600)) == 600


def test_explicit_value_overrides_source_size():
    assert resolve_canvas_size(1600, (518, 518)) == 1600


def test_process_one_passes_model_choice_through_to_cut_out(tmp_path):
    # Wiring check for the Fast/Quality toggle: process_one must forward
    # its model argument to cut_out unchanged. Mocked, not a real
    # inference call -- segment.cut_out's own behavior per model is
    # covered separately in tests/test_segment.py.
    source_path = tmp_path / "dummy.jpg"
    Image.new("RGB", (50, 50), (10, 20, 30)).save(source_path)
    fake_cutout = Image.new("RGBA", (50, 50), (10, 20, 30, 255))

    with patch("gembg.cli.cut_out", return_value=fake_cutout) as mock_cut_out, \
         patch("gembg.cli.compute_rotation", return_value=0.0), \
         patch("gembg.cli.mask_coverage_ratio", return_value=0.2):
        process_one(source_path, None, 0.08, model="fast")

    mock_cut_out.assert_called_once()
    assert mock_cut_out.call_args.kwargs.get("model") == "fast"
