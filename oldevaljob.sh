#!/bin/bash
#SBATCH -J navi
#SBATCH --mem 18000
#SBATCH --time 08:00:00
#SBATCH -p mid
#SBATCH --ntasks-per-node 4
#SBATCH --gres gpu:1

source /scratch/users/ocan13/.bash_profile

#hyper parameter seaerch
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f485_h60_e100 --seed 321 --oldvtest --vDev --filters 40 80 50 --hidden 60 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f485_h100_e100 --seed 321 --oldvtest --vDev --filters 40 80 50 --hidden 100 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f888_h100_e100 --seed 321 --oldvtest --vDev --filters 80 80 80 --hidden 100 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f646_h50_e160 --seed 321 --oldvtest --vDev --filters 60 40 60 --hidden 50 --embed 160
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f3105_h50_e120 --seed 321 --oldvtest --vDev --filters 30 100 50 --hidden 50 --embed 120
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f333_h50_e60 --seed 321 --oldvtest --vDev --filters 30 30 30 --hidden 50 --embed 60
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f3510_h50_e60 --seed 321 --oldvtest --vDev --filters 30 50 100 --hidden 50 --embed 60
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f454_h180_e100 --seed 321 --oldvtest --vDev --filters 40 50 40 --hidden 180 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f4154_h100_e100 --seed 321 --oldvtest --vDev --filters 40 150 40 --hidden 100 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f432_h50_e50 --seed 321 --oldvtest --vDev --filters 40 30 20 --hidden 50 --embed 50
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f1086_h100_e100 --seed 321 --oldvtest --vDev --filters 100 80 60 --hidden 100 --embed 100
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f6126_h120_e120 --seed 321 --oldvtest --vDev --filters 60 120 60 --hidden 120 --embed 120
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f666_h150_e150 --seed 321 --oldvtest --vDev --filters 60 60 60 --hidden 150 --embed 150
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f353_h50_e80 --seed 321 --oldvtest --vDev --filters 30 50 30 --hidden 50 --embed 80
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f844_h200_e120 --seed 321 --oldvtest --vDev --filters 80 60 40 --hidden 200 --embed 120
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f48_h120_e120_w15_512 --seed 321 --oldvtest --vDev --filters 40 80 --hidden 120 --embed 120 --window1 1 5 --window2 5 12
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f66_h120_e120_w15_512 --seed 321 --oldvtest --vDev --filters 60 60 --hidden 120 --embed 120 --window1 1 5 --window2 5 12
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f126_h120_e120_w15_512 --seed 321 --oldvtest --vDev --filters 120 60 --hidden 120 --embed 120 --window1 1 5 --window2 5 12
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f126_h120_e120_w15_515 --seed 321 --oldvtest --vDev --filters 120 60 --hidden 120 --embed 120 --window1 1 5 --window2 5 15
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f55_h60_e100_w15_515 --seed 321 --oldvtest --vDev --filters 50 50 --hidden 60 --embed 100 --window1 1 5 --window2 5 15
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f55_h60_e100_w17_514 --seed 321 --oldvtest --vDev --filters 50 50 --hidden 60 --embed 100 --window1 1 5 --window2 7 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f36_h80_e80_w17_514 --seed 321 --oldvtest --vDev --filters 30 60 --hidden 80 --embed 80 --window1 1 5 --window2 7 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f155_h120_e120_w17_514 --seed 321 --oldvtest --vDev --filters 150 50 --hidden 120 --embed 120 --window1 1 5 --window2 7 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f1055_h120_e120_w11_15_512 --seed 321 --oldvtest --vDev --filters 100 50 50 --hidden 120 --embed 120 --window1 1 1 5 --window2 1 5 12
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f355_h200_e120_w11_15_512 --seed 321 --oldvtest --vDev --filters 30 50 50 --hidden 200 --embed 120 --window1 1 1 5 --window2 1 5 12
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f355_h120_e120_w11_15_516 --seed 321 --oldvtest --vDev --filters 30 50 50 --hidden 120 --embed 120 --window1 1 1 5 --window2 1 5 16
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f358_h100_e100_w11_15_514 --seed 321 --oldvtest --vDev --filters 30 50 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f888_h100_e100_w11_15_514 --seed 321 --oldvtest --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f4512_h100_e200_w11_15_514 --seed 321 --oldvtest --vDev --filters 40 50 120 --hidden 100 --embed 200 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_2_f888_h100_e100_w11_15_514 --seed 322 --oldvtest --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f888_h120_e120_w11_15_514 --seed 321 --oldvtest --vDev --filters 80 80 80 --hidden 120 --embed 120 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_1 --log logs/sail_oldvtest_1_f101010_h120_e120_w11_15_514 --seed 321 --oldvtest --vDev --filters 100 100 100 --hidden 120 --embed 120 --window1 1 1 5 --window2 1 5 14

#L
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_1 --log logs/sail_oldvtest_l_1_f888_h100_e100_w11_15_514 --seed 321 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_2 --log logs/sail_oldvtest_l_2_f888_h100_e100_w11_15_514 --seed 322 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_3 --log logs/sail_oldvtest_l_3_f888_h100_e100_w11_15_514 --seed 323 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_4 --log logs/sail_oldvtest_l_4_f888_h100_e100_w11_15_514 --seed 324 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_5 --log logs/sail_oldvtest_l_5_f888_h100_e100_w11_15_514 --seed 325 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_6 --log logs/sail_oldvtest_l_6_f888_h100_e100_w11_15_514 --seed 326 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_7 --log logs/sail_oldvtest_l_7_f888_h100_e100_w11_15_514 --seed 327 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_8 --log logs/sail_oldvtest_l_8_f888_h100_e100_w11_15_514 --seed 328 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
#julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_9 --log logs/sail_oldvtest_l_9_f888_h100_e100_w11_15_514 --seed 329 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
julia mainflex.jl --epoch 100 --save mbank/sail_vtest_l_10 --log logs/sail_oldvtest_10_f888_h100_e100_w11_15_514 --seed 330 --oldvtest 1 --vDev --filters 80 80 80 --hidden 100 --embed 100 --window1 1 1 5 --window2 1 5 14
