#!/bin/bash
#$ -N navirl
#$ -q all.q@parcore-6-0
#$ -cwd
#$ -S /bin/bash
#$ -l gpu=1
##$ -l h_rt=48:00:00

julia hyperopt.jl --epoch 50 --maxevals 100 > logs/hyperopt.log
#rm navirl.*
