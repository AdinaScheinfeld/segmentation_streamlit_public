# /home/ads4015/segmentation_streamlit_public/segmentation_samples_streamlit_app.py - Streamlit app to show segmentation prediction samples for human eval

# --- Setup ---

# imports
import argparse
from collections import Counter
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import hashlib
from io import BytesIO
import json
import pandas as pd
from pathlib import Path
import random
import re
import requests
import streamlit as st


# --- Variable Definitions ---

MODELS = ["image_clip", "unet", "microsam"]
LABELS = ["A", "B", "C"]
PRETTY_DATATYPE = {
    "amyloid_plaque": "Amyloid beta plaque",
    "c_fos_positive": "cFos positive cells",
    "cell_nucleus": "Cell nucleus",
    "vessels": "Vessels",
}

# map model name -> URL column name in the CSV
MODEL_TO_URLCOL = {
    "image_clip": "pred_image_clip_url",
    "unet": "pred_unet_url",
    "microsam": "pred_microsam_url",
}

RESPONSES_TAB = "Responses"

EXPECTED_HEADER = [
    "response_id",
    "timestamp",
    "user_id",
    "sample_id",
    "datatype",
    "z",
    "rankA",
    "rankB",
    "rankC",
    "map_A",
    "map_B",
    "map_C",
]


# --- Helper Functions ---

# cached HTTP session for requests
@st.cache_resource(show_spinner=False)
def _http_session():
    s = requests.Session()
    return s


def get_or_create_worksheet(sh, title: str, n_rows: int = 2000, n_cols: int = 20):
    """
    Return worksheet `title`. If it doesn't exist, create it.
    """
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=n_rows, cols=n_cols)


def ensure_header(ws, header):
    """
    Ensure row 1 is the expected header. If sheet empty, write it.
    If header mismatch, warn (don't overwrite).
    """
    try:
        first = ws.row_values(1)
        if first == []:
            ws.append_row(header)
        elif first != header:
            st.warning(
                f"'{ws.title}' header row does not match expected schema. "
                "Logging may be misaligned."
            )
    except Exception as e:
        st.error(f"Failed to initialize Google Sheet tab '{ws.title}': {e}")
        st.stop()


def _normalize_drive_url(url: str) -> str:
    """
    Accepts any of these:
      - https://drive.google.com/file/d/<ID>/view?...
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>...
    Returns a direct download URL that serves image bytes:
      - https://drive.google.com/uc?export=download&id=<ID>
    """
    url = (url or "").strip()
    if not url:
        return ""
    # Already a direct "uc" style
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # /file/d/<ID>/...
    m = re.search(r"/file/d/([^/]+)/", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # If it's already some other public URL, return as-is
    return url

@st.cache_data(show_spinner=False)
def _fetch_image_bytes(url: str) -> bytes:
    """
    Fetch image bytes server-side. Works around Drive redirects and "view pages".
    """
    if not url:
        return b""
    direct = _normalize_drive_url(url)
    s = _http_session()
    r = s.get(direct, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.content


def show_image_url(url: str, title: str):
    # Always coerce to string (prevents '0' / NaN weirdness)
    url = "" if url is None else str(url).strip()
    if not url or url.lower() in {"nan", "none", "0", "0.0"}:
        st.error(f"Missing URL for: {title}")
        return
    try:
        img_bytes = _fetch_image_bytes(url)
        if not img_bytes:
            st.error(f"Empty image bytes for: {title}")
            st.caption(url)
            return
        st.image(BytesIO(img_bytes), caption=title, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load {title}: {e}")
        st.caption(url)


def show_image_url_cached(sample_key: str, url: str, title: str):
    """
    Cache image bytes in session_state so UI interactions (e.g., checkbox) don't
    cause re-downloads for the same sample.
    """
    url = "" if url is None else str(url).strip()
    if not url or url.lower() in {"nan", "none", "0", "0.0"}:
        st.error(f"Missing URL for: {title}")
        return

    if "img_bytes_cache" not in st.session_state:
        st.session_state.img_bytes_cache = {}

    cache_key = f"{sample_key}::{title}::{url}"
    if cache_key not in st.session_state.img_bytes_cache:
        # This uses the global disk/memo cache too, but also keeps it in RAM for this user session.
        st.session_state.img_bytes_cache[cache_key] = _fetch_image_bytes(url)

    img_bytes = st.session_state.img_bytes_cache.get(cache_key, b"")
    if not img_bytes:
        st.error(f"Empty image bytes for: {title}")
        st.caption(url)
        return
    st.image(BytesIO(img_bytes), caption=title, use_container_width=True)


# function to get deterministic mapping of labels to models per sample
def deterministic_mapping(sample_id: str, seed: int):
    h = hashlib.md5(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:8], 16))
    models = MODELS.copy()
    rng.shuffle(models)
    return dict(zip(LABELS, models))


@st.cache_resource(show_spinner=False)
def get_gsheet_client(_service_account_info: dict):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(_service_account_info, scopes=scopes)
    return gspread.authorize(creds)


# --- Main App ---

# main function
def main():

    # parse args
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=Path, default=Path("segmentation_samples_urls.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("segmentation_results.json"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--user_id", type=str, default="anon")
    args = ap.parse_args()

    # set random seed
    random.seed(args.seed)

    # Important: keep URL columns as strings (avoid NaN -> float -> 0/0.0 issues)
    data_csv = args.data_csv
    if not data_csv.exists():
        # On Streamlit Cloud, relative paths are from repo root; also handle "streamlit/" subfolder layouts.
        data_csv = Path(__file__).resolve().parent / args.data_csv.name
    df = pd.read_csv(data_csv, dtype=str, keep_default_na=False)

    # --------------------
    # UI: reduce top padding + tighten layout (avoid scrolling)
    # --------------------
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
          /* Keep enough top padding so the Streamlit header never clips the title */
           .block-container {
            padding-top: 1.8rem;   /* enough to avoid Streamlit header overlap */
            padding-bottom: 0.6rem;
            }

            h1 {
            margin-top: 0.2rem !important;
            font-size: 2.05rem !important;  /* slightly smaller */
            line-height: 1.25 !important;
            }

          /* Tighten vertical spacing globally */
          div[data-testid="stVerticalBlock"] { gap: 0.15rem; }

          /* Tighten caption spacing under title and under images */
          .stCaption { margin-top: 0.0rem !important; margin-bottom: 0.15rem !important; }

          /* Slightly smaller images so everything fits on one screen */
          .stImage img { max-height: 235px; object-fit: contain; }

          /* Compact checkbox / widgets spacing */
          div[data-testid="stCheckbox"] { margin-top: 0.15rem; margin-bottom: 0.15rem; }

          /* Compact radio spacing (keep stacked, just reduce padding) */
          div[role="radiogroup"] > label { padding: 0.0rem 0.2rem; }

          /* Reduce extra whitespace around the form container */
          div[data-testid="stForm"] { padding-top: 0.0rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # --------------------
    # CONNECT TO GOOGLE SHEETS (via Streamlit secrets)
    # --------------------
    try:
        gsheet_id = st.secrets["GSHEET_ID"]
        service_account_info = st.secrets["GCP_SERVICE_ACCOUNT"]

        gc = get_gsheet_client(service_account_info)
        sh = gc.open_by_key(gsheet_id)
        ws = get_or_create_worksheet(sh, RESPONSES_TAB)
        ensure_header(ws, EXPECTED_HEADER)
    except Exception as e:
        st.error("Failed to connect to Google Sheets. Check Streamlit secrets + sheet sharing.")
        st.exception(e)
        st.stop()

    # app title
    st.title("Segmentation Prediction Ranking")

    # instructions to display in app
    st.caption("Rank A/B/C best → worst (no ties). Model identities are hidden.")

    # --------------------
    # SCREEN 0: ENTER RATER ID (before showing any samples)
    # --------------------
    if "user_id" not in st.session_state:
        st.session_state.user_id = args.user_id
    if "started" not in st.session_state:
        st.session_state.started = False

    if not st.session_state.started:
        st.markdown("## Welcome")
        st.write("Enter your rater ID to begin. (This can be initials or a short name.)")

        st.session_state.user_id = st.text_input(
            "Rater ID",
            value=st.session_state.user_id,
        ).strip()

        cA, cB = st.columns([1, 3])
        with cA:
            start_clicked = st.button("Start")

        if start_clicked:
            if not st.session_state.user_id:
                st.error("Please enter a rater ID.")
                st.stop()
            st.session_state.started = True
            st.rerun()
        st.stop()

    # session state init
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "results" not in st.session_state:
        st.session_state.results = []

    if st.session_state.idx >= len(df):
        st.success("Done — thank you!")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # If user provided "something.json", convert to "something_<user>_<stamp>.json"
        out_path = args.out_json
        if out_path.suffix.lower() == ".json":
            out_path = out_path.with_name(f"{out_path.stem}_{st.session_state.user_id}_{stamp}.json")
        else:
            # If they pass a directory or no suffix, create a file inside it
            out_path = out_path / f"segmentation_eval_results_{st.session_state.user_id}_{stamp}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.write("Saved results to:", str(out_path))
        st.stop()

    # get current row
    row = df.iloc[st.session_state.idx]
    sample_key = str(row.sample_id)  # stable key for this sample

    # --------------------
    # STABLE RANDOMIZE (per slice, deterministic)
    # --------------------
    if "mappings" not in st.session_state:
        st.session_state.mappings = {}

    if st.session_state.idx not in st.session_state.mappings:
        st.session_state.mappings[st.session_state.idx] = deterministic_mapping(
            str(row.sample_id), args.seed
        )

    mapping = st.session_state.mappings[st.session_state.idx]


    # --------------------
    # DISPLAY IMAGE + PREDS
    # --------------------

    # Header row: empty GT column | inline header + slice counter
    _h_gt, _h_main = st.columns([1, 4])
    with _h_main:
        pretty_type = PRETTY_DATATYPE.get(row["datatype"], row["datatype"])

        st.markdown(
            f"""
            <div style="margin-bottom:0.15rem;">
            <div style="display:flex; align-items:baseline; gap:0.75rem;">
                <h3 style="margin:0;">
                Reference image + predictions
                </h3>
                <h3 style="margin:0; font-weight:400; color:#555;">
                (Slice {st.session_state.idx + 1} / {len(df)})
                </h3>
            </div>
            <div style="margin-top:0.1rem; font-size:0.95rem; color:#444;">
                Patch type: <strong>{pretty_type}</strong>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # layout: GT | Image | A | B | C
    col_gt, col_img, c1, c2, c3 = st.columns([1, 1, 1, 1, 1])

    show_gt = st.checkbox("Show ground truth", key=f"show_gt_{st.session_state.idx}")

    with col_gt:
        if show_gt: # if checked, show GT
            show_image_url_cached(sample_key, row["gt_url"], "Ground Truth")
        else:
            # Placeholder keeps column width stable
            st.empty()

    with col_img:
        show_image_url_cached(sample_key, row["image_url"], "Image")

    for col, label in zip([c1, c2, c3], LABELS):
        model = mapping[label]
        url_col = MODEL_TO_URLCOL[model]
        pred_url = row[url_col]

        with col:
            show_image_url_cached(sample_key, pred_url, f"Prediction {label}")

    # ---- Prefetch next sample (warm caches) ----
    if "prefetched_idx" not in st.session_state:
        st.session_state.prefetched_idx = -1

    if st.session_state.prefetched_idx != st.session_state.idx:
        st.session_state.prefetched_idx = st.session_state.idx
        next_idx = st.session_state.idx + 1
        if next_idx < len(df):
            next_row = df.iloc[next_idx]
            try:
                _ = _fetch_image_bytes(next_row["image_url"])
                _ = _fetch_image_bytes(next_row["pred_image_clip_url"])
                _ = _fetch_image_bytes(next_row["pred_unet_url"])
                _ = _fetch_image_bytes(next_row["pred_microsam_url"])
            except Exception:
                pass


    # st.markdown("### Rank each prediction (no ties allowed)")
    rank_options = ["Best", "Middle", "Worst"]

    with st.form(key=f"rank_form_{st.session_state.idx}", clear_on_submit=False):
        rankA = st.radio("Prediction A", rank_options, key=f"rankA_{st.session_state.idx}", horizontal=True)
        rankB = st.radio("Prediction B", rank_options, key=f"rankB_{st.session_state.idx}", horizontal=True)
        rankC = st.radio("Prediction C", rank_options, key=f"rankC_{st.session_state.idx}", horizontal=True)

        submitted = st.form_submit_button("Next")

    label_to_rank = {"A": rankA, "B": rankB, "C": rankC}

    if submitted:

        # Enforce: exactly one Best, one Middle, one Worst (no ties)
        counts = Counter(label_to_rank.values())
        ok = (counts.get("Best", 0) == 1 and
              counts.get("Middle", 0) == 1 and
              counts.get("Worst", 0) == 1)
        if not ok:
            st.error("No ties allowed: assign exactly one Best, one Middle, and one Worst across A/B/C.")
            st.stop()

        # Append to Google Sheet (one row per sample rating)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response_id = f"{st.session_state.user_id}::{row.sample_id}"
        try:
            ws.append_row([
                response_id,
                stamp,
                st.session_state.user_id,
                str(row.sample_id),
                str(row.datatype),
                str(int(row.z)),
                rankA,
                rankB,
                rankC,
                mapping["A"],
                mapping["B"],
                mapping["C"],
            ])
        except Exception as e:
            st.error(f"Failed to log to Google Sheets: {e}")
            st.stop()

        # store result
        st.session_state.results.append({
            "user_id": st.session_state.user_id,
            "sample_id": row.sample_id,
            "datatype": row.datatype,
            "z": int(row.z),

            # per-pred label rank (no ties allowed)
            "ranking_labels": label_to_rank,

            # also store numeric form for convenience
            "ranking_numeric": {k: {"Best": 1, "Middle": 2, "Worst": 3}[v] for k, v in label_to_rank.items()},

            "model_map": mapping,  # A/B/C -> actual model name
        })
        st.session_state.idx += 1
        st.rerun()



# main app entry point
if __name__ == "__main__":
    main()








