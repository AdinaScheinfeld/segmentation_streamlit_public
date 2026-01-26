# /home/ads4015/segmentation_streamlit_public/build_segmentation_samples_list.py - build CSV listing segmentation prediction samples for human eval

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import re


# --- Variable Definitions ---

# regex to extract cvfold from runfolder name
NTR_RE  = re.compile(r"cvfold(\d+)_ntr(\d+)_")
PRED_ID_RE = re.compile(r"(patch_\d+)_(vol\d+)_ch(\d+)_pred_")

# map datatype to finetune patches subdir
DTYPE_TO_PATCHDIR = {
    "amyloid_plaque": "amyloid_plaque_patches",
    "c_fos_positive": "c_fos_positive_patches",
    "cell_nucleus": "cell_nucleus_patches",
    "vessels": "vessels_patches",
}

# Short dirs exist for super_sweep2 and unet_random2
DTYPE_TO_SHORTDIR = {
    "amyloid_plaque": "amyloid_plaque",
    "c_fos_positive": "c_fos_positive",
    "cell_nucleus": "cell_nucleus",
    "vessels": "vessels",
}

# map method -> datatype subdir under root/preds/
# NOTE:
#   - unet_image_clip (super_sweep2) uses short dirs (amyloid_plaque, vessels, ...)
#   - unet_random and microsam use *_patches dirs
DTYPE_TO_METHOD_SUBDIR = {
    "unet_image_clip": DTYPE_TO_SHORTDIR,
    "unet_random": DTYPE_TO_SHORTDIR,
    "microsam": DTYPE_TO_PATCHDIR,
}

# map model to datatype subdir under preds/
METHOD_TO_PREDS_SUBDIR = {
    "unet_image_clip": "preds", # .../<runfolder>/preds/*.nii.gz
    "unet_random": "preds", # .../<runfolder>/preds/*.nii.gz
    "microsam": "patches" # .../<runfolder>/patches/*.nii.gz
}


# --- Helper Functions ---

# function to parse patch/vol/ch from prediction filename
def parse_pred_ids(pred_filename: str):
    m = PRED_ID_RE.search(pred_filename)
    if not m:
        raise ValueError(f"Could not parse patch/vol/ch from: {pred_filename}")
    return m.group(1), m.group(2), int(m.group(3))


def pred_key_from_filename(pred_filename: str, method: str):
    """
    Return a join key shared across methods: (patch_id, vol_id, ch)
    """
    # microsam_b2 now follows the same *_pred_* filename convention as the UNets
    return parse_pred_ids(pred_filename)

def resolve_dtype_base(root: Path, datatype: str, method: str) -> Path:
    dtype_dir = DTYPE_TO_METHOD_SUBDIR[method][datatype]
    return root / "preds" / dtype_dir


# function to list runfolders for a given datatype
def list_runfolders(root: Path, datatype: str, method: str):
    base = resolve_dtype_base(root, datatype, method)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


# # function to pick one runfolder per fold deterministically
# def pick_one_runfolder_per_fold_max_ntr_with_info(runfolders):
#     """
#     For each fold, choose the runfolder with the highest ntr.
#     Returns:
#       picked: dict fold_id -> Path(runfolder)
#       info:   dict fold_id -> int(ntr)
#     """
#     best = {}  # fold -> (ntr, name, Path)
#     for rf in runfolders:
#         m = NTR_RE.search(rf.name)
#         if not m:
#             continue
#         fold = int(m.group(1))
#         ntr = int(m.group(2))
#         key = (ntr, rf.name)  # tie-break by name
#         if fold not in best or key > (best[fold][0], best[fold][1]):
#             best[fold] = (ntr, rf.name, rf)

#     picked = {fold: tpl[2] for fold, tpl in best.items()}
#     info   = {fold: tpl[0] for fold, tpl in best.items()}
#     return picked, info

# function to pick one runfolder per fold deterministically
def runfolder_matches_size(rf_name: str, method: str, size: int) -> bool:
    """
    All three methods now use runfolders that match cvfold*_ntr{size}_...
    """
    if method in ("unet_image_clip", "unet_random", "microsam"):
        m = NTR_RE.search(rf_name)
        return (m is not None) and (int(m.group(2)) == size)
    return False


# function to list prediction files in a runfolder
def list_pred_files(runfolder: Path, method: str):
    sub = METHOD_TO_PREDS_SUBDIR[method]
    pred_dir = runfolder if sub == "" else (runfolder / sub)
    if not pred_dir.exists():
        return []
    return sorted(pred_dir.glob("*_pred_*.nii.gz"))  # ONLY preds, not probs


# # function to get corresponding runfolder for other models
# def corresponding_runfolder(model_root: Path, datatype: str, runfolder_name: str, method: str):
#     dtype_dir = DTYPE_TO_METHOD_SUBDIR[method][datatype]
#     return model_root / "preds" / dtype_dir / runfolder_name

def build_pred_index_for_size(root: Path, datatype: str, method: str, size: int):
    """
    Build mapping: (patch_id, vol_id, ch) -> (pred_path, runfolder_name, pred_filename)
    across ALL runfolders matching the requested size.
    Deterministic: sorted runfolders, sorted files; keep first occurrence.
    """
    idx = {}
    base = resolve_dtype_base(root, datatype, method)
    runfolders = list_runfolders(root, datatype, method=method)
    runfolders = [rf for rf in runfolders if runfolder_matches_size(rf.name, method, size)]

    for rf in runfolders:
        files = list_pred_files(rf, method=method)
        for p in files:
            try:
                k = pred_key_from_filename(p.name, method)
            except Exception:
                # skip weird files that don't follow naming conventions
                continue
            if k not in idx:
                idx[k] = (p, rf.name, p.name)
    return idx


# function to check if a slice has any foreground in ANY prediction
def slice_has_any_foreground(pred_paths, z: int) -> bool:
    """
    Returns True iff ANY of the prediction volumes has any nonzero voxel at slice z.
    Uses nibabel slicing to avoid loading the full 96^3 volume into RAM.
    """
    for p in pred_paths:
        img = nib.load(str(p))
        sl = np.asanyarray(img.dataobj[:, :, z])
        if np.any(sl > 0):
            return True
    return False


# function to build candidate z list
def candidate_z_list(z_targets, z_dim: int, z_border: int):
    """
    Build a deterministic list of z indices:
      1) z_targets (in given order), filtered to valid range
      2) remaining z in a center-out order (deterministic), excluding borders
    """
    lo = z_border
    hi = z_dim - z_border
    valid = [z for z in z_targets if lo <= z < hi]

    center = (lo + hi - 1) / 2.0
    remaining = [z for z in range(lo, hi) if z not in set(valid)]
    remaining.sort(key=lambda z: (abs(z - center), z))  # deterministic
    return valid + remaining


# function to select z slices for predictions
def select_z_slices_for_pred(pred_paths, z_targets, slices_per_pred: int, z_border: int):
    """
    For a given pred volume triplet (clip/only/rand), pick exactly slices_per_pred
    distinct z indices such that at least one model is nonzero on that slice.
    Returns:
        z_list: list of selected z indices
        info:   list of tuples (z, is_preferred, has_fg)
    """
    # infer z dimension from first path
    img0 = nib.load(str(pred_paths[0]))
    z_dim = img0.shape[2]

    cand = candidate_z_list(z_targets, z_dim=z_dim, z_border=z_border)
    chosen = []
    info = []
    chosen_set = set()

    for z in cand:
        if z in chosen_set:
            continue

        has_fg = slice_has_any_foreground(pred_paths, z)
        is_pref = z in z_targets
        if has_fg:
            chosen.append(z)
            chosen_set.add(z)
            info.append((z, is_pref, has_fg))
            if len(chosen) == slices_per_pred:
                return chosen, info

        elif is_pref:
            # record preferred but empty slices for logging
            info.append((z, is_pref, has_fg))

    return [], info  # couldn't find enough


# --- Main Function ---

def main():

    # argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet_image_clip_root", type=Path, required=True, help="Path to .../temp_selma_segmentation_preds_super_sweep2")
    ap.add_argument("--unet_random_root", type=Path, required=True, help="Path to .../temp_selma_segmentation_preds_unet_random2")
    ap.add_argument("--microsam_root", type=Path, required=True, help="Path to .../compare_methods/micro_sam/finetuned_cross_val_b")
    ap.add_argument("--finetune_patches_root", type=Path, required=True, help="Path to selma3d_finetune_patches (contains *_patches subfolders)")
    ap.add_argument("--datatypes", nargs="+", default=["amyloid_plaque", "cell_nucleus", "vessels"], help="List of datatypes to include")
    ap.add_argument("--train_sizes", nargs="+", type=int, default=[5, 15], help="Sizes to include (ntr for UNet; pool for microsam).")
    ap.add_argument("--z_planes", nargs="+", type=int, default=[32, 64], help="Preferred z indices; script will backfill if empty.")
    ap.add_argument("--slices_per_pred", type=int, default=2, help="How many z slices to pick per pred volume (distinct).")
    ap.add_argument("--z_border", type=int, default=1, help="Exclude z in [0,z_border) and [z_dim-z_border, z_dim).")
    ap.add_argument("--preds_per_size", type=int, default=2, help="Number of patch-volumes to select per datatype per size (after intersecting methods)")
    ap.add_argument("--out_csv", type=Path, default=Path("segmentation_samples.csv"), help="Output CSV file path")
    args = ap.parse_args()

    # roots dict
    roots = {
        "unet_image_clip": args.unet_image_clip_root,
        "unet_random": args.unet_random_root,
        "microsam": args.microsam_root,
    }

    # list to hold all records
    records = []

    for datatype in args.datatypes:
        print(f"\nProcessing datatype: {datatype}", flush=True)

        for size in args.train_sizes:
            print(f"  Selecting for size={size} (ntr{size})", flush=True)

            idx_clip = build_pred_index_for_size(roots["unet_image_clip"], datatype, "unet_image_clip", size)
            idx_rand = build_pred_index_for_size(roots["unet_random"], datatype, "unet_random", size)
            idx_ms   = build_pred_index_for_size(roots["microsam"], datatype, "microsam", size)

            # ---- EARLY DEBUG: INDEX CONTENTS ----
            def _dbg_idx(name, idx):
                keys = sorted(idx.keys())
                print(f"    [DEBUG INDEX] {name}: {len(keys)} preds", flush=True)
                for k in keys[:5]:
                    p, rf, fname = idx[k]
                    print(f"      key={k}  file={fname}  (runfolder={rf})", flush=True)
                if len(keys) > 5:
                    print(f"      ... ({len(keys) - 5} more)", flush=True)

            print(
                f"\n  [DEBUG] datatype={datatype} size={size}",
                flush=True,
            )

            _dbg_idx("unet_image_clip", idx_clip)
            _dbg_idx("unet_random", idx_rand)
            _dbg_idx("microsam", idx_ms)

            common_keys = sorted(set(idx_clip) & set(idx_rand) & set(idx_ms))
            print(f"    [DEBUG INTERSECTION] common keys = {len(common_keys)}", flush=True)

            if len(common_keys) < args.preds_per_size:
                raise SystemExit(
                    f"Not enough common pred keys for datatype={datatype}, size={size}. "
                    f"Need {args.preds_per_size}, found {len(common_keys)}."
                )

            # Pick exactly 2 keys per fold (3 folds -> 6 total) in a deterministic way
            def _fold_from_runfolder(rf_name: str) -> int:
                # runfolder like: cvfold2_ntr15_...
                m = re.search(r"^cvfold(\d+)_", rf_name)
                if not m:
                    raise ValueError(f"Could not parse fold from runfolder: {rf_name}")
                return int(m.group(1))

            keys_by_fold = {0: [], 1: [], 2: []}
            for k in common_keys:
                clip_path, clip_rf, _ = idx_clip[k]
                f = _fold_from_runfolder(clip_rf)
                if f in keys_by_fold:
                    keys_by_fold[f].append(k)

            picked_keys = []
            for f in sorted(keys_by_fold):
                keys_by_fold[f].sort()
                picked_keys.extend(keys_by_fold[f][:2])  # 2 test images per fold

            if len(picked_keys) < 6:
                raise SystemExit(
                    f"Not enough per-fold keys for datatype={datatype}, size={size}. "
                    f"Need 6 (2 per fold), found {len(picked_keys)}."
                )

            for k in picked_keys:
                clip_path, clip_rf, clip_fname = idx_clip[k]
                rand_path, rand_rf, rand_fname = idx_rand[k]
                ms_path,   ms_rf,   ms_fname   = idx_ms[k]

                # ---- PRINT WHERE THIS PRED IS COMING FROM ----
                print(
                    "    [PRED SOURCE] "
                    f"datatype={datatype} size={size} key={k}\n"
                    f"      unet_image_clip: runfolder={clip_rf}\n"
                    f"        path={clip_path}\n"
                    f"        file={clip_fname}\n"
                    f"      unet_random:     runfolder={rand_rf}\n"
                    f"        path={rand_path}\n"
                    f"        file={rand_fname}\n"
                    f"      microsam:        runfolder={ms_rf}\n"
                    f"        path={ms_path}\n"
                    f"        file={ms_fname}",
                    flush=True,
                )

                pred_paths = [clip_path, rand_path, ms_path]
                z_list, z_info = select_z_slices_for_pred(
                    pred_paths,
                    z_targets=args.z_planes,
                    slices_per_pred=args.slices_per_pred,
                    z_border=args.z_border,
                )
                if len(z_list) != args.slices_per_pred:
                    raise SystemExit(
                        f"Could not find {args.slices_per_pred} non-empty slices for "
                        f"{datatype} size={size} k={k}"
                    )

                patch_id, vol_id, ch = k  # k = (patch_id, vol_id, ch)
                print(f"    [{datatype} | size{size} | {patch_id}_{vol_id}_ch{ch}] "
                      f"(clip={clip_rf}, rand={rand_rf}, ms={ms_rf})", flush=True)

                for z, is_pref, has_fg in z_info:
                    if is_pref and has_fg:
                        print(f"      preferred z={z} ✓", flush=True)
                    elif is_pref and not has_fg:
                        print(f"      preferred z={z} ✗ (empty)", flush=True)
                    elif has_fg:
                        print(f"      backfilled z={z} ✓", flush=True)

                patchdir = args.finetune_patches_root / DTYPE_TO_PATCHDIR[datatype]
                image_path = patchdir / f"{patch_id}_{vol_id}_ch{ch}.nii.gz"
                gt_path    = patchdir / f"{patch_id}_{vol_id}_ch{ch}_label.nii.gz"
                if not image_path.exists() or not gt_path.exists():
                    raise SystemExit(f"Missing image/gt for {datatype} {patch_id}_{vol_id}_ch{ch}")

                for z in z_list:
                    # use a stable stem based on the key (since filenames differ across methods)
                    sample_stem = f"{patch_id}_{vol_id}_ch{ch}"
                    sample_id = f"{datatype}_size{size}_{sample_stem}_z{z}"
                    records.append({
                        "sample_id": sample_id,
                        "datatype": datatype,
                        "train_size": int(size),
                        # keep one representative filename for bookkeeping (clip fname)
                        "filename": clip_fname,
                        "z": int(z),
                        "image_path": str(image_path),
                        "gt_path": str(gt_path),
                        "unet_image_clip_path": str(clip_path),
                        "unet_random_path": str(rand_path),
                        "microsam_path": str(ms_path),
                        "unet_image_clip_runfolder": clip_rf,
                        "unet_random_runfolder": rand_rf,
                        "microsam_runfolder": ms_rf,
                    })

        per_dtype = len(args.train_sizes) * args.preds_per_size * args.slices_per_pred
        print(f"  Added {per_dtype} rows for {datatype} "
              f"({len(args.train_sizes)} sizes × {args.preds_per_size} preds × {args.slices_per_pred} slices_per_pred)", flush=True)

    # save to csv
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=100).reset_index(drop=True)  # shuffle rows to randomize order deterministically

    # If we pick 2 test images per fold, total preds per (datatype,size) = 2 * n_folds
    # Default to 3 folds if present; infer from what we actually logged.
    n_folds = df["unet_image_clip_runfolder"].str.extract(r"^cvfold(\d+)_")[0].nunique()
    preds_per_dtype_size = 2 * int(n_folds)
    expected = len(args.datatypes) * len(args.train_sizes) * preds_per_dtype_size * args.slices_per_pred
    print(f"\nTotal samples collected: {len(df)}", flush=True)
    print(f"[DEBUG EXPECTED] n_folds={n_folds}, preds_per_dtype_size={preds_per_dtype_size}, expected={expected}", flush=True)
    if len(df) != expected:
        raise SystemExit(f"ERROR: expected {expected} rows but got {len(df)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}", flush=True)

if __name__ == "__main__":
    main()



