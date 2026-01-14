# /home/ads4015/segmentation_streamlit_public/build_segmentation_samples_list.py - build CSV listing segmentation prediction samples for human eval

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
import re


# --- Variable Definitions ---

# regex to extract cvfold from runfolder name
NTR_RE = re.compile(r"cvfold(\d+)_ntr(\d+)_")
PRED_ID_RE = re.compile(r"(patch_\d+)_(vol\d+)_ch(\d+)_pred_")

# map datatype to finetune patches subdir
DTYPE_TO_PATCHDIR = {
    "amyloid_plaque": "amyloid_plaque_patches",
    "c_fos_positive": "c_fos_positive_patches",
    "cell_nucleus": "cell_nucleus_patches",
    "vessels": "vessels_patches",
}

# map model to method subdir
DTYPE_TO_METHOD_SUBDIR = {

    # image+clip model uses shorter names
    "image_clip": {
        "amyloid_plaque": "amyloid_plaque",
        "c_fos_positive": "c_fos_positive",
        "cell_nucleus": "cell_nucleus",
        "vessels": "vessels",
    },

    # unet and microsam use *_patches folder names
    "unet": {
        "amyloid_plaque": "amyloid_plaque_patches",
        "c_fos_positive": "c_fos_positive_patches",
        "cell_nucleus": "cell_nucleus_patches",
        "vessels": "vessels_patches",
    },
    "microsam": {
        "amyloid_plaque": "amyloid_plaque_patches",
        "c_fos_positive": "c_fos_positive_patches",
        "cell_nucleus": "cell_nucleus_patches",
        "vessels": "vessels_patches",
    },
}

# map model to datatype subdir under preds/
METHOD_TO_PREDS_SUBDIR = {
    "image_clip": "preds", # .../<runfolder>/preds/*.nii.gz
    "unet": "", # .../<runfolder>/*.nii.gz
    "microsam": "patches" # .../<runfolder>/patches/*.nii.gz
}


# --- Helper Functions ---

# function to parse patch/vol/ch from prediction filename
def parse_pred_ids(pred_filename: str):
    m = PRED_ID_RE.search(pred_filename)
    if not m:
        raise ValueError(f"Could not parse patch/vol/ch from: {pred_filename}")
    return m.group(1), m.group(2), int(m.group(3))


# function to list runfolders for a given datatype
def list_runfolders(root: Path, datatype: str, method: str):
    dtype_dir = DTYPE_TO_METHOD_SUBDIR[method][datatype]
    base = root / "preds" / dtype_dir
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


# function to pick one runfolder per fold deterministically
def pick_one_runfolder_per_fold_max_ntr_with_info(runfolders):
    """
    For each fold, choose the runfolder with the highest ntr.
    Returns:
      picked: dict fold_id -> Path(runfolder)
      info:   dict fold_id -> int(ntr)
    """
    best = {}  # fold -> (ntr, name, Path)
    for rf in runfolders:
        m = NTR_RE.search(rf.name)
        if not m:
            continue
        fold = int(m.group(1))
        ntr = int(m.group(2))
        key = (ntr, rf.name)  # tie-break by name
        if fold not in best or key > (best[fold][0], best[fold][1]):
            best[fold] = (ntr, rf.name, rf)

    picked = {fold: tpl[2] for fold, tpl in best.items()}
    info   = {fold: tpl[0] for fold, tpl in best.items()}
    return picked, info


# function to list prediction files in a runfolder
def list_pred_files(runfolder: Path, method: str):
    sub = METHOD_TO_PREDS_SUBDIR[method]
    pred_dir = runfolder if sub == "" else (runfolder / sub)
    if not pred_dir.exists():
        return []
    return sorted(pred_dir.glob("*_pred_*.nii.gz"))  # ONLY preds, not probs


# function to get corresponding runfolder for other models
def corresponding_runfolder(model_root: Path, datatype: str, runfolder_name: str, method: str):
    dtype_dir = DTYPE_TO_METHOD_SUBDIR[method][datatype]
    return model_root / "preds" / dtype_dir / runfolder_name


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
    ap.add_argument("--image_clip_root", type=Path, required=True, help="Path to segmentation preds from image+clip model")
    ap.add_argument("--unet_root", type=Path, required=True, help="Path to segmentation preds from unet model")
    ap.add_argument("--microsam_root", type=Path, required=True, help="Path to segmentation preds from microsam model")
    ap.add_argument("--finetune_patches_root", type=Path, required=True, help="Path to selma3d_finetune_patches (contains *_patches subfolders)")
    ap.add_argument("--datatypes", nargs="+", default=["amyloid_plaque", "c_fos_positive", "cell_nucleus", "vessels"], help="List of datatypes to include")
    ap.add_argument("--z_planes", nargs="+", type=int, default=[32, 64], help="Preferred z indices; script will backfill if empty.")
    ap.add_argument("--slices_per_pred", type=int, default=2, help="How many z slices to pick per pred volume (distinct).")
    ap.add_argument("--z_border", type=int, default=1, help="Exclude z in [0,z_border) and [z_dim-z_border, z_dim).")
    ap.add_argument("--preds_per_fold", type=int, default=2, help="Number of predictions to select per fold")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2], help="List of folds to include")
    ap.add_argument("--out_csv", type=Path, default=Path("segmentation_samples.csv"), help="Output CSV file path")
    args = ap.parse_args()

    # roots dict
    roots = {
        "image_clip": args.image_clip_root,
        "unet": args.unet_root,
        "microsam": args.microsam_root,
    }

    # list to hold all records
    records = []

    for datatype in args.datatypes:
        print(f"\nProcessing datatype: {datatype}", flush=True)

        # 1) Find runfolders in image_clip root
        runfolders = list_runfolders(roots["image_clip"], datatype, method="image_clip")
        if not runfolders:
            raise SystemExit(f"No runfolders found for datatype={datatype} under {roots['image_clip']}")

        # 2) Pick ONE runfolder per fold deterministically
        picked, picked_ntr = pick_one_runfolder_per_fold_max_ntr_with_info(runfolders)

        # Print which ntr was chosen per fold (in args.folds order)
        chosen_msg = ", ".join([f"cvfold{f}: ntr{picked_ntr.get(f, 'NA')}" for f in args.folds])
        print(f"  Chosen runfolders by fold (max ntr): {chosen_msg}", flush=True)

        # Ensure we have desired folds
        missing_folds = [f for f in args.folds if f not in picked]
        if missing_folds:
            raise SystemExit(f"Missing folds {missing_folds} for datatype={datatype} in {roots['image_clip']}")

        for fold in args.folds:
            rf_clip = picked[fold]
            run_name = rf_clip.name

            # 3) Ensure corresponding runfolders exist for other models
            rf_unet = corresponding_runfolder(roots["unet"], datatype, run_name, method="unet")
            rf_microsam = corresponding_runfolder(roots["microsam"], datatype, run_name, method="microsam")

            if not rf_unet.exists():
                raise SystemExit(f"Missing unet runfolder: {rf_unet}")
            if not rf_microsam.exists():
                raise SystemExit(f"Missing microsam runfolder: {rf_microsam}")

            # 4) Pick exactly preds_per_fold files from the image_clip runfolder
            clip_files = list_pred_files(rf_clip, method="image_clip")
            if len(clip_files) < args.preds_per_fold:
                raise SystemExit(f"Found only {len(clip_files)} preds in {rf_clip}/preds, expected {args.preds_per_fold}")

            # debugging counters
            dbg = {
                "total_clip_files": 0,
                "missing_unet_file": 0,
                "missing_microsam_file": 0,
                "insufficient_slices": 0,
                "missing_image_or_gt": 0,
                "selected_preds": 0,
            }

            # We need exactly preds_per_fold pred-volumes that each yield slices_per_pred valid z's.
            selected_pred_vols = 0
            for clip_path in clip_files:
                dbg["total_clip_files"] += 1
                if selected_pred_vols >= args.preds_per_fold:
                    break

                unet_dir = rf_unet if METHOD_TO_PREDS_SUBDIR["unet"] == "" else (rf_unet / METHOD_TO_PREDS_SUBDIR["unet"])
                microsam_dir = rf_microsam if METHOD_TO_PREDS_SUBDIR["microsam"] == "" else (rf_microsam / METHOD_TO_PREDS_SUBDIR["microsam"])

                unet_path = unet_dir / clip_path.name
                microsam_path = microsam_dir / clip_path.name

                if not unet_path.exists():
                    dbg["missing_unet_file"] += 1
                    continue
                if not microsam_path.exists():
                    dbg["missing_microsam_file"] += 1
                    continue

                pred_paths = [clip_path, unet_path, microsam_path]
                
                z_list, z_info = select_z_slices_for_pred(
                    pred_paths,
                    z_targets=args.z_planes,
                    slices_per_pred=args.slices_per_pred,
                    z_border=args.z_border
                )
                if len(z_list) != args.slices_per_pred:
                    dbg["insufficient_slices"] += 1

                # ---- PRINT SELECTION DETAILS ----
                patch_id, vol_id, ch = parse_pred_ids(clip_path.name)
                print(f"  [{datatype} | cvfold{fold} | {patch_id}_{vol_id}_ch{ch}]", flush=True)

                for z, is_pref, has_fg in z_info:
                    if is_pref and has_fg:
                        print(f"    preferred z={z} ✓", flush=True)
                    elif is_pref and not has_fg:
                        print(f"    preferred z={z} ✗ (empty)", flush=True)
                    elif has_fg:
                        print(f"    backfilled z={z} ✓", flush=True)

                patchdir = args.finetune_patches_root / DTYPE_TO_PATCHDIR[datatype]
                image_path = patchdir / f"{patch_id}_{vol_id}_ch{ch}.nii.gz"
                gt_path    = patchdir / f"{patch_id}_{vol_id}_ch{ch}_label.nii.gz"
                if not image_path.exists() or not gt_path.exists():
                    dbg["missing_image_or_gt"] += 1
                    continue

                # Append one row per chosen z (unique by construction)
                for z in z_list:
                    sample_id = f"{datatype}_{run_name}_{clip_path.stem}_z{z}"
                    records.append({
                        "sample_id": sample_id,
                        "datatype": datatype,
                        "cvfold": fold,
                        "runfolder": run_name,
                        "filename": clip_path.name,
                        "z": int(z),
                        "image_path": str(image_path),
                        "gt_path": str(gt_path),
                        "image_clip_path": str(clip_path),
                        "unet_path": str(unet_path),
                        "microsam_path": str(microsam_path),
                    })

                selected_pred_vols += 1
                dbg['selected_preds'] += 1

            # print debug summary for this fold
            print(f'   [DEBUG {datatype} fold{fold}] {dbg}', flush=True)

            if selected_pred_vols != args.preds_per_fold:
                raise SystemExit(
                    f"Could not find {args.preds_per_fold} pred-volumes with "
                    f"{args.slices_per_pred} non-empty slices each for datatype={datatype}, fold={fold} "
                    f"in runfolder={run_name}. Found {selected_pred_vols}."
                )

        # helpful logging
        per_dtype = len(args.folds) * args.preds_per_fold * args.slices_per_pred
        print(f"  Added {per_dtype} rows for {datatype} "
              f"({len(args.folds)} folds × {args.preds_per_fold} preds × {args.slices_per_pred} slices_per_pred)", flush=True)

    # save to csv
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=100).reset_index(drop=True)  # shuffle rows to randomize order deterministically
    expected = len(args.datatypes) * len(args.folds) * args.preds_per_fold * args.slices_per_pred
    print(f"\nTotal samples collected: {len(df)}", flush=True)
    if len(df) != expected:
        raise SystemExit(f"ERROR: expected {expected} rows but got {len(df)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}", flush=True)

if __name__ == "__main__":
    main()



