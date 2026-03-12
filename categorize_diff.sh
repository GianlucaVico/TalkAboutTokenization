#!/bin/bash
#SBATCH -o diff.err
#SBATCH -e diff.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH -p gpu-troja
#SBATCH --gpus 2
#SBATCH -C gpuram95G
#SBATCH -J llama_judge

# ./categorize_diff.py
./llm_job.py
