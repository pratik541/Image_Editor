from pathlib import Path

import streamlit as st

from gembg.cli import SUPPORTED_EXTENSIONS, process_one

st.set_page_config(page_title="Gem photo cleanup", page_icon=":material/diamond:")

st.title("Gem photo cleanup")
st.caption(
    "Remove the background and shadow, straighten elongated cuts, and place each "
    "photo on a white canvas."
)

with st.form("process_folder", border=True):
    input_dir = st.text_input(
        "Input folder", value="input", placeholder="Path to folder with source photos"
    )
    output_dir = st.text_input(
        "Output folder", value="output", placeholder="Path to write cleaned photos into"
    )
    with st.container(horizontal=True):
        canvas_size = st.number_input(
            "Canvas size (px)", min_value=200, max_value=4000, value=1600, step=100
        )
        margin = st.slider("Margin", min_value=0.0, max_value=0.3, value=0.08, step=0.01)
    submitted = st.form_submit_button(
        "Process folder", icon=":material/play_arrow:", type="primary"
    )

results = st.container()

if submitted:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        results.error(f"Input folder not found: {input_path}")
        st.stop()

    source_files = sorted(
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_files:
        results.warning("No supported image files (.jpg, .jpeg, .png) found in that folder.")
        st.stop()

    output_path.mkdir(parents=True, exist_ok=True)

    with results:
        progress = st.progress(0.0, text="Starting…")
        needs_review = []
        previews = []

        for i, source_file in enumerate(source_files):
            progress.progress(
                i / len(source_files),
                text=f"Processing {source_file.name} ({i + 1}/{len(source_files)})",
            )
            output_image, review_flag = process_one(source_file, canvas_size, margin)
            output_image.save(output_path / source_file.name, quality=95)
            if review_flag:
                needs_review.append(source_file.name)
            previews.append((source_file.name, output_image))

        progress.progress(1.0, text="Done")

        st.success(f"Processed {len(source_files)} photo(s) into `{output_path}`")

        if needs_review:
            st.warning(
                "Segmentation looked uncertain for these files, so they were not "
                "auto-rotated (background/shadow removal still applied) — worth a "
                "manual look: " + ", ".join(needs_review)
            )

        st.subheader("Preview")
        gallery = previews[:12]
        columns = st.columns(4)
        for idx, (name, image) in enumerate(gallery):
            with columns[idx % 4]:
                st.image(image, caption=name)
        if len(previews) > len(gallery):
            st.caption(f"+{len(previews) - len(gallery)} more saved to {output_path}")
