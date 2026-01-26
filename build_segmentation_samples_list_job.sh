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
source activate monai-env2

# run the script to build the segmentation samples list
python -u /home/ads4015/segmentation_streamlit_public/build_segmentation_samples_list.py \
  --unet_image_clip_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_super_sweep2 \
  --unet_random_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2 \
  --microsam_root /midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val_b2 \
  --finetune_patches_root /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches \
  --datatypes amyloid_plaque cell_nucleus vessels \
  --train_sizes 5 15 \
  --preds_per_size 2 \
  --z_planes 32 64 \
  --slices_per_pred 2 \
  --z_border 2 \
  --out_csv /midtier/paetzollab/scratch/ads4015/ssl_streamlit/segmentation_samples_list.csv


# indicate ending
echo "Finished build_segmentation_samples_list_job.sh at $(date)"




