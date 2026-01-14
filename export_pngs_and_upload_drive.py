# /home/ads4015/segmentation_streamlit_public/export_pngs_and_upload_drive.py - script to export PNGs and upload to Google Drive


import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import nibabel as nib

from PIL import Image

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# -------------------------
# Helpers: image conversion
# -------------------------
def load_slice(path_str: str, z: int) -> np.ndarray:
    img = nib.load(path_str)
    sl = np.asanyarray(img.dataobj[:, :, z])
    return sl

def to_uint8(img2d: np.ndarray, is_mask: bool) -> np.ndarray:
    """Convert 2D float/int array to uint8 for PNG."""
    arr = np.asarray(img2d)
    if is_mask:
        # masks/preds: assume 0/1 or probabilities; visualize as 0/255
        return (arr > 0).astype(np.uint8) * 255

    # intensity image: robust normalize (avoid outliers)
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255).astype(np.uint8)

def save_png(img_u8: np.ndarray, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(out_png)


# -------------------------
# Helpers: Google Drive
# -------------------------
def drive_login(creds_path: Path, client_secrets_path: Path) -> GoogleDrive:
    gauth = GoogleAuth()
    # Ensure it knows where client secrets are (sometimes needed on refresh)
    gauth.settings["client_config_file"] = str(client_secrets_path)

    gauth.LoadCredentialsFile(str(creds_path))
    if gauth.credentials is None:
        raise RuntimeError(f"No credentials loaded from {creds_path}")
    if gauth.access_token_expired:
        gauth.Refresh()
        gauth.SaveCredentialsFile(str(creds_path))
    else:
        gauth.Authorize()

    return GoogleDrive(gauth)

def ensure_folder(drive: GoogleDrive, name: str, parent_id: str | None) -> str:
    """Return folder id for (name) under parent_id, creating it if needed."""
    if parent_id:
        q = f"'{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder' and title='{name}'"
    else:
        q = f"trashed=false and mimeType='application/vnd.google-apps.folder' and title='{name}'"

    found = drive.ListFile({"q": q}).GetList()
    if found:
        return found[0]["id"]

    meta = {"title": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [{"id": parent_id}]
    folder = drive.CreateFile(meta)
    folder.Upload()
    return folder["id"]

def upload_file(drive: GoogleDrive, local_path: Path, parent_id: str) -> str:
    f = drive.CreateFile({"title": local_path.name, "parents": [{"id": parent_id}]})
    f.SetContentFile(str(local_path))
    f.Upload()

    # make it publicly readable
    f.InsertPermission({
        "type": "anyone",
        "value": "anyone",
        "role": "reader",
    })

    return f["id"]

def direct_uc_url(file_id: str) -> str:
    # direct image URL that works with st.image
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True, help="Local export directory (PNGs saved here)")
    ap.add_argument("--out_csv", type=Path, required=True, help="Output CSV with Drive URLs")
    ap.add_argument("--creds_json", type=Path, required=True)
    ap.add_argument("--client_secrets_json", type=Path, required=True)
    ap.add_argument("--drive_root_name", type=str, default="seg_eval_assets", help="Top-level Drive folder name")
    ap.add_argument("--overwrite_local", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.samples_csv)

    # Login + make folder structure
    drive = drive_login(args.creds_json, args.client_secrets_json)
    root_id = ensure_folder(drive, args.drive_root_name, parent_id=None)

    # per-sample unique dir name sanitizer
    safe_re = re.compile(r"[^A-Za-z0-9_\-]+")

    out_rows = []

    for i, row in df.iterrows():
        sample_id = str(row["sample_id"])
        datatype = str(row["datatype"])
        z = int(row["z"])

        # Drive folders: root/datatype/sample_id/
        dtype_id = ensure_folder(drive, datatype, parent_id=root_id)
        sample_folder_name = safe_re.sub("_", sample_id)[:120]
        sample_id_drive = ensure_folder(drive, sample_folder_name, parent_id=dtype_id)

        # Local folder: out_dir/datatype/sample_id/
        sample_local = args.out_dir / datatype / sample_folder_name
        sample_local.mkdir(parents=True, exist_ok=True)

        # Export + upload these files
        items = [
            ("image", row["image_path"], False),
            ("gt", row["gt_path"], True),
            ("pred_image_clip", row["image_clip_path"], True),
            ("pred_unet", row["unet_path"], True),
            ("pred_microsam", row["microsam_path"], True),
        ]

        url_map = {}
        for key, nii_path, is_mask in items:
            png_path = sample_local / f"{key}_z{z:02d}.png"
            if png_path.exists() and not args.overwrite_local:
                # still upload (or skip upload if you want). We'll upload to ensure Drive has it.
                pass

            sl = load_slice(str(nii_path), z)
            u8 = to_uint8(sl, is_mask=is_mask)
            save_png(u8, png_path)

            fid = upload_file(drive, png_path, parent_id=sample_id_drive)
            url_map[f"{key}_url"] = direct_uc_url(fid)

        out_rows.append({
            "sample_id": sample_id,
            "datatype": datatype,
            "z": z,
            # URLs for Streamlit Cloud
            **url_map,
            # keep original bookkeeping if you want
            "cvfold": row.get("cvfold", ""),
            "runfolder": row.get("runfolder", ""),
            "filename": row.get("filename", ""),
        })

        if (i + 1) % 10 == 0:
            print(f"Uploaded {i+1}/{len(df)} samples...", flush=True)

    out_df = pd.DataFrame(out_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved URL CSV:", args.out_csv, flush=True)
    print("Drive folder name:", args.drive_root_name, flush=True)

if __name__ == "__main__":
    main()
