#!/bin/bash
#$ -J navirl
#SBATCH --mem 10024
#SBATCH --time 01:00:00
#SBATCH -p ai_gpu
#SBATCH --gres gpu:1
#SBATCH --time 48:00:00
#SBATCH --nodelist cn4

julia mainflex.jl --epoch 100 --save mbank/slurm --log logs/slurm
