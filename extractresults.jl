using ArgParse, StatsBase, Logging


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        ("--prefix"; help = "prefix"; default = "")
        ("--log"; help = "log file"; default = "")
        ("--nummodels"; arg_type=Int; default=3; help="number of models")
    end
    return parse_args(s)
end		

args = parse_commandline()

function main()
    args["log"] != "" && Logging.configure(filename=args["log"])
    Logging.configure(level=INFO)
    
    info("*** Parameters ***")
    for k in keys(args); info("$k -> $(args[k])"); end

    s = [874, 1293, 1070] ./ 3237
    p = [224, 242, 236] ./ 702

    s = WeightVec(s);
    p = WeightVec(p);

    singles = zeros(args["nummodels"]*2, 3)
    paragraphs = zeros(args["nummodels"]*2, 3)

    for i=1:args["nummodels"]
        for j=1:2
            for (ind,map) in [(1, "grid"), (2, "jelly"), (3, "l")]
                fname = string(args["prefix"], "_",i,"_",j,"_",map)
                lines = readlines(fname)
                last = split(lines[end])
                singles[(i-1)*2+j, ind] = float(last[10])
                paragraphs[(i-1)*2+j, ind] = float(last[14])
            end
        end
    end

    info("**Beam Singles**")
    info(singles)

    info("**Beam Paragraphs**")
    info(paragraphs)

    info("Avg Single: $(mean_and_std(mean(singles, 1), s))")
    info("Avg Paragraph: $(mean_and_std(mean(paragraphs, 1), p))")

    singles = zeros(2, 3)
    paragraphs = zeros(2, 3)

    for j=1:2
        for (ind,map) in [(1, "grid"), (2, "jelly"), (3, "l")]
            fname = string(args["prefix"], "_ens_",j,"_",map)
            lines = readlines(fname)
            last = split(lines[end])
            singles[j, ind] = float(last[10])
            paragraphs[j, ind] = float(last[14])
        end
    end
    
    info("Ensemble Avg Single: $(mean_and_std(mean(singles, 1), s))")
    info("Ensemble Avg Paragraph: $(mean_and_std(mean(paragraphs, 1), p))")

end

main()
