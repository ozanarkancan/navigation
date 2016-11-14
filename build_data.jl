#using ArgParse
include("util.jl")

function main()
	#=
	s = ArgParseSettings()

	@add_arg_table s begin
		"--bs"
		help = "batch size"
		default = 100
		arg_type = Int
	end

	args = parse_args(s)
	=#

	files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
	"data/instructions/SingleSentenceZeroInitial.jelly.json"]
	build_data(files, string("grid_jelly.jld"))

	files = ["data/instructions/SingleSentenceZeroInitial.grid.json",
	"data/instructions/SingleSentenceZeroInitial.l.json"]
	build_data(files, string("grid_l.jld"))

	files = ["data/instructions/SingleSentenceZeroInitial.l.json",
	"data/instructions/SingleSentenceZeroInitial.jelly.json"]
	build_data(files, string("l_jelly.jld"))
end

main()
