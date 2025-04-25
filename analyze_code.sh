#!/bin/bash

#SBATCH --job-name=charxiv_bench
#SBATCH --mail-user=jhsansom@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=60g
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=cse692w25_class
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%x-%j.log

module load cuda/12.1.1
source /scratch/cse692w25_class_root/cse692w25_class/jhsansom/miniconda3/etc/profile.d/conda.sh
conda activate vipergpt

#python benchmark_on_charxiv.py

export PYTHONPATH=$PYTHONPATH:.
python analyze_code.py