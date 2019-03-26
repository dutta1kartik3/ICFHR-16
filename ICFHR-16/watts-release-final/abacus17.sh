#!/bin/bash
#SBATCH -A cvit
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -p long
#SBATCH -t 480:00:00
#SBATCH --mem-per-cpu=3900
#SBATCH -C 96g

echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
export LD_LIBRARY_PATH=/home/cvit/kartik/watts-master/util/vlfeat-0.9.18/bin/glnxa64/:$LD_LIBRARY_PATH
/scratch/matlab/R2013b/bin/matlab -r 'main'

echo "Program finished with exit code $? at: `date`"
