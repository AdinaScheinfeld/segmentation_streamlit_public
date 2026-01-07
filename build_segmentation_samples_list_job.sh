#!/bin/bash
#SBATCH --job-name=build_seg_samples
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/build_seg_samples_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/build_seg_samples_%j.err


# /home/ads4015/segmentation_streamlit_public/build_segmentation_samples_list_job.sh - script to build segmentation samples list CSV

set -euo pipefail
mkdir -p /midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/

# indicate starting
echo "Starting build_segmentation_samples_list_job.sh at $(date)"


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# run the script to build the segmentation samples list
python -u /home/ads4015/segmentation_streamlit_public/build_segmentation_samples_list.py \
  --image_clip_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_autumn_sweep_27_v2 \
  --image_only_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_expert_sweep_31_v2 \
  --random_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_rand_v2 \
  --finetune_patches_root /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches \
  --datatypes amyloid_plaque c_fos_positive cell_nucleus vessels \
  --folds 0 1 2 \
  --preds_per_fold 2 \
  --z_planes 32 64 \
  --slices_per_pred 2 \
  --z_border 2 \
  --out_csv /midtier/paetzollab/scratch/ads4015/ssl_streamlit/segmentation_samples_list.csv


# indicate ending
echo "Finished build_segmentation_samples_list_job.sh at $(date)"




