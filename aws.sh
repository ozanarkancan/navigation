#Base
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t1_base.csv --numbatch 900 --limit --0.075 --log logs/hopt_t1_base --level debug --taskf turn_to_x --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t2_base.csv --numbatch 400 --limit --0.075 --log logs/hopt_t2_base --level debug --taskf move_to_x --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t3_base.csv --numbatch 1200 --limit --0.075 --log logs/hopt_t3_base --level debug --taskf combined_12 --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t4_base.csv --numbatch 1000 --limit --0.075 --log logs/hopt_t4_base --level debug --taskf turn_and_move_to_x --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t5_base.csv --numbatch 500 --limit --0.075 --log logs/hopt_t5_base --level debug --taskf lang_only --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t6_base.csv --numbatch 1500 --limit --0.075 --log logs/hopt_t6_base --level debug --taskf combined_1245 --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t7_base.csv --numbatch 1000 --limit --0.075 --log logs/hopt_t7_base --level debug --taskf move_until --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t8_base.csv --numbatch 2000 --limit --0.075 --log logs/hopt_t8_base --level debug --taskf any_combination --hopt
#julia pretraining.jl --wvecs --encoding multihot --savecsv learningcurves/hopt_t9_base.csv --numbatch 2500 --limit --0.075 --log logs/hopt_t9_base --level debug --taskf all_classes --hopt

#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t1_grid.csv --numbatch 800 --limit --0.075 --log logs/hopt_t1_grid --level debug --taskf turn_to_x --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t2_grid.csv --numbatch 400 --limit --0.075 --log logs/hopt_t2_grid --level debug --taskf move_to_x --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t3_grid.csv --numbatch 1100 --limit --0.075 --log logs/hopt_t3_grid --level debug --taskf combined_12 --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t4_grid.csv --numbatch 1000 --limit --0.075 --log logs/hopt_t4_grid --level debug --taskf turn_and_move_to_x --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t5_grid.csv --numbatch 450 --limit --0.075 --log logs/hopt_t5_grid --level debug --taskf lang_only --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t6_grid.csv --numbatch 1500 --limit --0.075 --log logs/hopt_t6_grid --level debug --taskf combined_1245 --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t7_grid.csv --numbatch 1000 --limit --0.075 --log logs/hopt_t7_grid --level debug --taskf move_until --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t8_grid.csv --numbatch 2000 --limit --0.075 --log logs/hopt_t9_grid --level debug --taskf any_combination --hopt
#julia pretraining.jl --wvecs --window1 1 5 --window2 20 1 --filters 50 5 --winit 5 --worldatt 100 --savecsv learningcurves/hopt_t9_grid.csv --numbatch 2500 --limit --0.075 --log logs/hopt_t10_grid --level debug --taskf all_classes --hopt
