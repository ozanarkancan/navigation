#!/bin/bash

mname="flex.jl"
oname="experiments2/sail"
lpath="mbank/sail"
#enc="multihot"
enc="grid"
N=10
H=120
cat=""

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_1_l_jelly.jld" --log $oname"_"$i"_1_grid" --test 5 
    loads=$loads" "$lpath"_"$i"_1_l_jelly.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_1_grid" --test 5

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_2_l_jelly.jld" --log $oname"_"$i"_2_grid" --test 6 
    loads=$loads" "$lpath"_"$i"_2_l_jelly.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_2_grid" --test 6

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_1_grid_l.jld" --log $oname"_"$i"_1_jelly" --test 3 
    loads=$loads" "$lpath"_"$i"_1_grid_l.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_1_jelly" --test 3

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_2_grid_l.jld" --log $oname"_"$i"_2_jelly" --test 4 
    loads=$loads" "$lpath"_"$i"_2_grid_l.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_2_jelly" --test 4

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_1_grid_jelly.jld" --log $oname"_"$i"_1_l" --test 1 
    loads=$loads" "$lpath"_"$i"_1_grid_jelly.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_1_l" --test 1

loads=""
for i in `seq 1 $N`;
do
    julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $lpath"_"$i"_2_grid_jelly.jld" --log $oname"_"$i"_2_l" --test 2 
    loads=$loads" "$lpath"_"$i"_2_grid_jelly.jld";
done

julia beamsearchflex.jl $preva $inpout --worldatt 100 --hidden $H --decdrops 0.5 0.9 --bs 1 --encoding $enc --model $mname --load $loads --log $oname"_ens_2_l" --test 2
