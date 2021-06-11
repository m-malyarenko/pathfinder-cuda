vim run.slurm#!/bin/bash
#SBATCH -J pathfinder
#SBATCH -N 1
#SBATCH -p tornado-k40
#SBATCH -o ./out/std/pathfinder-%j.out
#SBATCH -e ./out/err/pathfinder-%j.err
#SBATCH -t 00:10:00 

if [[ -f /etc/profile.d/modules-basis.sh ]]; then
source /etc/profile.d/modules-basis.sh
fi

module purge
module load nvidia/cuda/10.1
./build/bin/pathfinder $1
