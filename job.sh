#!/bin/bash
#$ -J navirl
#SBATCH --mem 10024
#SBATCH --time 48:00:00
#SBATCH -p all_gpu
#SBATCH --gres gpu:1
#SBATCH --time 48:00:00

#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_1 --seed 1
#julia mainflex.jl --epoch 100 --save mbank/sail_2 --log logs/sail_2 --seed 2
#julia mainflex.jl --epoch 100 --save mbank/sail_3 --log logs/sail_3 --seed 3
#julia mainflex.jl --epoch 100 --save mbank/sail_4 --log logs/sail_4 --seed 4
#julia mainflex.jl --epoch 100 --save mbank/sail_5 --log logs/sail_5 --seed 5
#julia mainflex.jl --epoch 100 --save mbank/sail_6 --log logs/sail_6 --seed 6
#julia mainflex.jl --epoch 100 --save mbank/sail_7 --log logs/sail_7 --seed 7
#julia mainflex.jl --epoch 100 --save mbank/sail_8 --log logs/sail_8 --seed 8
#julia mainflex.jl --epoch 100 --save mbank/sail_9 --log logs/sail_9 --seed 9
julia mainflex.jl --epoch 100 --save mbank/sail_10 --log logs/sail_10 --seed 10
