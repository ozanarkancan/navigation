#!/bin/bash
#$ -N navirl
#$ -q ai.q@ahtapot-5-1
#$ -cwd
#$ -S /bin/bash
#$ -l gpu=1
##$ -l h_rt=48:00:00

julia main.jl --epoch 400 --hidden 64 --embed 64 --window 15 --filters 500 --model model09.jl --testfile "data/instructions/SingleSentenceZeroInitial.grid.json" --trainfile "l_jelly.jld" > model9_h64_w15_f500_l_jelly.log
#rm navirl.*
