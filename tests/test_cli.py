from gembg.cli import resolve_canvas_size


def test_none_matches_larger_side_of_source_image():
    assert resolve_canvas_size(None, (518, 518)) == 518
    assert resolve_canvas_size(None, (600, 400)) == 600
    assert resolve_canvas_size(None, (400, 600)) == 600


def test_explicit_value_overrides_source_size():
    assert resolve_canvas_size(1600, (518, 518)) == 1600
