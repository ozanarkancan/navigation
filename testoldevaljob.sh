#!/bin/bash
#SBATCH -J navi
#SBATCH --mem 18000
#SBATCH --time 08:00:00
#SBATCH -p mid
#SBATCH --ntasks-per-node 4
#SBATCH --gres gpu:1

source /scratch/users/ocan13/.bash_profile

mname="flex.jl"
oname="experiments/sail_vtest_l_"
lpath="mbank/sail_vtest_l_"
lpathsuffix="_vtest_grid_jelly"
#enc="multihot"
enc="grid"
N=10
H=100
cat=""

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $inpout --worldatt 100 --hidden $H --bs 1 --encoding $enc --model $mname --load $lpath$i$lpathsuffix".jld" --log $oname$i$lpathsuffix --oldvtest 1
    loads=$loads" "$lpath$i$lpathsuffix".jld";
done

julia beamsearchflex.jl $inpout --worldatt 100 --hidden $H --bs 1 --encoding $enc --model $mname --load $loads --log $oname"ens"$lpathsuffix --oldvtest 1

