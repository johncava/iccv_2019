#!/bin/bash

#SBATCH -p gpu                      # Use gpu partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue

#SBATCH --gres=gpu:1                # Request two GPUs

#SBATCH -t 0-24:00                  # wall time (D-HH:MM)
#SBATCH -o gcn_L6.out             # STDOUT (%j = JobId)
#SBATCH -e gcn_L6.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # send-to address

source activate /home/jcava/.conda/envs/pytorch-1.20-gpu/
python gcn_L6.py
conda deactivate