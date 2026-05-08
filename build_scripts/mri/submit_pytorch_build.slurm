#!/bin/bash
#SBATCH --job-name=pytorch-build
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --constraint=cuda
#SBATCH --output=pytorch_build_%j.out
#SBATCH --error=pytorch_build_%j.err

################################################################################
# PyTorch Build Batch Job
#
# Submit with: sbatch build_scripts/mri/submit_pytorch_build.slurm
# Monitor with: tail -f pytorch_build_<jobid>.out
################################################################################

echo "========================================"
echo "PyTorch Build Job Started"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# Run the build script with --clean
cd ~/osu-hidl
bash build_scripts/mri/build_pytorch_mri.sh --clean

echo ""
echo "========================================"
echo "Build Job Completed"
echo "========================================"
echo "End time: $(date)"
