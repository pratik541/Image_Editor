import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import streamlit as st

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
IMAGE_TYPES = ["jpg", "jpeg", "png"]
IMPORT_TIMEOUT_SECONDS = 90
PROCESS_TIMEOUT_SECONDS = 60


def _log(message):
    # flush=True so this shows up immediately in a hosting platform's log
    # viewer even if stdout would otherwise be buffered.
    print(f"[gembg-app] {message}", file=sys.stderr, flush=True)


def _load_process_one():
    _log("import: starting `from gembg.cli import process_one`")
    t0 = time.time()
    from gembg.cli import process_one  # deferred: heavy ML imports

    _log(f"import: finished in {time.time() - t0:.1f}s")
    return process_one

st.set_page_config(page_title="Gem photo cleanup", page_icon=":material/diamond:")

st.title("Gem photo cleanup")
st.caption(
    "Remove the background and shadow, straighten elongated cuts, and place each "
    "photo on a white canvas."
)

st.session_state.setdefault("uploader_version", 0)

with st.container(horizontal=True):
    upload_mode = st.segmented_control(
        "Add photos as",
        ["Files", "Folder"],
        default="Folder",
        required=True,
        key="upload_mode",
    )
    if st.button("Clear selection", icon=":material/clear_all:"):
        st.session_state.uploader_version += 1
        st.session_state.pop("last_run", None)
        st.rerun()

uploader_key = f"uploaded_{upload_mode}_{st.session_state.uploader_version}"

with st.form("process_folder", border=True):
    if upload_mode == "Files":
        uploaded_files = st.file_uploader(
            "Photos",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            help="Select one or more gemstone photos from your device.",
            key=uploader_key,
        )
    else:
        uploaded_files = st.file_uploader(
            "Photo folder",
            type=IMAGE_TYPES,
            accept_multiple_files="directory",
            help="Select a folder of gemstone photos from your device.",
            key=uploader_key,
        )
    with st.container(horizontal=True):
        custom_canvas = st.checkbox("Custom canvas size")
        canvas_size = None
        if custom_canvas:
            canvas_size = st.number_input(
                "Canvas size (px)", min_value=200, max_value=4000, value=1600, step=100
            )
        margin = st.slider("Margin", min_value=0.0, max_value=0.3, value=0.08, step=0.01)
    if not custom_canvas:
        st.caption("Output size matches each photo's own size unless overridden above.")
    submitted = st.form_submit_button(
        "Process", icon=":material/play_arrow:", type="primary"
    )

results = st.container()

if submitted:
    _log(f"form submitted with {len(uploaded_files) if uploaded_files else 0} file(s)")

    if not uploaded_files:
        results.warning("Choose at least one .jpg, .jpeg, or .png photo first.")
        st.stop()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with results:
        # Run the heavy import (and each image) in a worker thread with a
        # hard timeout, so a stalled dependency (e.g. onnxruntime/numba
        # failing to initialize on a constrained host) surfaces as a clear
        # error instead of an infinite silent spinner.
        executor = ThreadPoolExecutor(max_workers=1)

        with st.spinner("Loading the segmentation model (first run only)…"):
            try:
                process_one = executor.submit(_load_process_one).result(
                    timeout=IMPORT_TIMEOUT_SECONDS
                )
            except FutureTimeoutError:
                st.error(
                    f"Loading the background-removal model timed out after "
                    f"{IMPORT_TIMEOUT_SECONDS}s. This usually means this hosting "
                    "environment can't initialize one of its dependencies "
                    "(onnxruntime, or numba/llvmlite pulled in by rembg) -- check "
                    "the app's server logs for anything logged right around now."
                )
                st.stop()

        progress = st.progress(0.0, text="Starting…")
        needs_review = []
        previews = []

        for i, uploaded_file in enumerate(uploaded_files):
            file_name = Path(uploaded_file.name).name
            progress.progress(
                i / len(uploaded_files),
                text=f"Processing {file_name} ({i + 1}/{len(uploaded_files)})",
            )
            _log(f"processing ({i + 1}/{len(uploaded_files)}): starting {file_name}")
            t0 = time.time()
            try:
                output_image, review_flag = executor.submit(
                    process_one, uploaded_file, canvas_size, margin
                ).result(timeout=PROCESS_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                _log(f"processing: TIMED OUT on {file_name} after {time.time() - t0:.1f}s")
                st.warning(f"{file_name} timed out after {PROCESS_TIMEOUT_SECONDS}s -- skipped.")
                needs_review.append(f"{file_name} (timed out)")
                continue
            _log(f"processing: finished {file_name} in {time.time() - t0:.1f}s")
            output_image.save(OUTPUT_DIR / file_name, quality=95)
            if review_flag:
                needs_review.append(file_name)
            previews.append((file_name, output_image))

        progress.progress(1.0, text="Done")
        _log("all files processed")

    st.session_state["last_run"] = {
        "total": len(uploaded_files),
        "needs_review": needs_review,
        "previews": previews,
    }

last_run = st.session_state.get("last_run")
if last_run:
    with results:
        st.success(f"Processed {last_run['total']} photo(s)")

        if last_run["needs_review"]:
            st.warning(
                "Segmentation looked uncertain for these files, so they were not "
                "auto-rotated (background/shadow removal still applied) — worth a "
                "manual look: " + ", ".join(last_run["needs_review"])
            )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, image in last_run["previews"]:
                image_bytes = io.BytesIO()
                is_png = Path(name).suffix.lower() == ".png"
                image.save(image_bytes, format="PNG" if is_png else "JPEG", quality=95)
                zip_file.writestr(name, image_bytes.getvalue())

        st.download_button(
            "Download all as ZIP",
            data=zip_buffer.getvalue(),
            file_name="cleaned_photos.zip",
            mime="application/zip",
            icon=":material/download:",
        )

        st.subheader("Preview")
        gallery = last_run["previews"][:12]
        columns = st.columns(4)
        for idx, (name, image) in enumerate(gallery):
            with columns[idx % 4]:
                st.image(image, caption=name)
        if len(last_run["previews"]) > len(gallery):
            st.caption(
                f"+{len(last_run['previews']) - len(gallery)} more — download the ZIP "
                "above to get all of them"
            )
