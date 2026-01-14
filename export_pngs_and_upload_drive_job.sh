#!/bin/bash
#SBATCH --job-name=seg_png_drive
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/export_png_drive_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/export_png_drive_%j.err

# /home/ads4015/segmentation_streamlit_public/export_pngs_and_upload_drive_job.sh - script to export PNGs and upload to Google Drive

set -euo pipefail
mkdir -p /midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs

module load anaconda3/2022.10-34zllqw
source activate gdrive-env2

# indicate starting
echo "Starting export_pngs_and_upload_drive_job.sh at $(date)"

# define variables
SAMPLES_CSV=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/segmentation_samples_list.csv # path to samples CSV generated using build_segmentation_samples_list_job.sh
OUT_DIR=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/seg_eval_export_pngs # local output dir for exported PNGs
OUT_CSV=/home/ads4015/segmentation_streamlit_public/segmentation_samples_urls.csv # path to output CSV with Google Drive URLs

# run the export and upload script
python -u /home/ads4015/segmentation_streamlit_public/export_pngs_and_upload_drive.py \
  --samples_csv "${SAMPLES_CSV}" \
  --out_dir "${OUT_DIR}" \
  --out_csv "${OUT_CSV}" \
  --creds_json /home/ads4015/segmentation_streamlit_public/gdrive_creds.json \
  --client_secrets_json /home/ads4015/segmentation_streamlit_public/client_secrets.json \
  --drive_root_name "seg_eval_assets" \
  --overwrite_local

# indicate ending
echo "Finished export_pngs_and_upload_drive_job.sh at $(date)"
echo "Output CSV with Google Drive URLs: ${OUT_CSV}"
