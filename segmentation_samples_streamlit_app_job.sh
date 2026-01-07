#!/bin/bash
#SBATCH --job-name=seg_rank_ui
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/streamlit_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs/streamlit_%j.err


# /home/ads4015/segmentation_streamlit_public/segmentation_samples_streamlit_app_job.sh - script to run Streamlit app for segmentation sample ranking

set -euo pipefail
mkdir -p /midtier/paetzollab/scratch/ads4015/ssl_streamlit/logs

# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate gdrive-env

# set Streamlit port
PORT=8501

echo "Running Streamlit on host: $(hostname)"
echo "Port: ${PORT}"
echo "If port is busy, change PORT to something else like 8502."

# run the Streamlit app
# IMPORTANT: streamlit flags must come BEFORE the `--` that separates app args
python -m streamlit run /home/ads4015/ssl_project/streamlit/segmentation_samples_streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT} \
  --server.headless true \
  -- \
  --data_csv /home/ads4015/ssl_project/streamlit/segmentation_samples_urls.csv \
  --out_json /midtier/paetzollab/scratch/ads4015/ssl_streamlit/segmentation_samples_results.json \
  --seed 100 \
  --user_id anon



