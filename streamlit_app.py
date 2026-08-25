from pathlib import Path

import streamlit as st

from gembg.cli import process_one

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

st.set_page_config(page_title="Gem photo cleanup", page_icon=":material/diamond:")

st.title("Gem photo cleanup")
st.caption(
    "Remove the background and shadow, straighten elongated cuts, and place each "
    "photo on a white canvas."
)

with st.form("process_folder", border=True):
    uploaded_files = st.file_uploader(
        "Photo folder",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files="directory",
        help="Select a folder of gemstone photos from your device.",
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
    if not uploaded_files:
        results.warning("Choose a folder with .jpg, .jpeg, or .png photos first.")
        st.stop()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with results:
        progress = st.progress(0.0, text="Starting…")
        needs_review = []
        previews = []

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = Path(uploaded_file.name).name
            progress.progress(
                i / len(uploaded_files),
                text=f"Processing {file_name} ({i + 1}/{len(uploaded_files)})",
            )
            output_image, review_flag = process_one(uploaded_file, canvas_size, margin)
            output_image.save(OUTPUT_DIR / file_name, quality=95)
            if review_flag:
                needs_review.append(file_name)
            previews.append((file_name, output_image))

        progress.progress(1.0, text="Done")

        st.success(f"Processed {len(uploaded_files)} photo(s) into `{OUTPUT_DIR}`")

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
            st.caption(f"+{len(previews) - len(gallery)} more saved to {OUTPUT_DIR}")
