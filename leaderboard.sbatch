#!/bin/bash
#SBATCH --job-name=ptfs-leaderboard
#SBATCH --partition=singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=leaderboard-%j.txt
#SBATCH --export=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    echo "icpx is not in PATH. Load the course Intel compiler environment before submitting this job."
    exit 127
fi

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Job: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Compiler version:"
icpx --version

CXX=icpx make
./test

FIRST="OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES"
SECOND="--ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores"

printf 'Leaderboard command: ./leaderboard.sh "%s" "%s"\n' "$FIRST" "$SECOND"
./leaderboard.sh "$FIRST" "$SECOND"
