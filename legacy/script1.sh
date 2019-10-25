#!/bin/bash
 
#SBATCH -n 4                        # number of cores
#SBATCH -t 0-24:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # send-to address

module load python/3.6.4

python interpol_v4.py
