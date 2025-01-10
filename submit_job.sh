#!/bin/bash
#SBATCH --job-name=TAD        # Job name
#SBATCH --output=log_files/TAD.out      # Standard output and error log
#SBATCH --error=log_files/TAD.err       # Separate file for error logs
#SBATCH --partition=gpu72           # Partition for GPU nodes (qgpu)
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Number of tasks per node
#SBATCH --cpus-per-task=16           # Number of CPU cores per task (use the full node)
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --time=72:00:00              # Set time limit to 72 hours (3 days)
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications for job begin, end, and failure
#SBATCH --mail-user=ZXiao@walton.uark.edu  # Your email address


module load os/el7 gcc/11.2.1 python/miniforge-24.3.0
conda init
conda activate cuda_env
cd ~/Research/TAD/
python -m w2v_culture.step5_score
