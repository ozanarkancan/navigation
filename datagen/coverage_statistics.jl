using Logging, DataFrames, ProgressMeter, ArgParse

include("data_generator.jl")

function coverage(taskf, unmatched, freq; limit=1e4)
    matched = Set()
    patience = 0
    ind = 0
    patlim = string(taskf) == "lang_only" ? 5e5 : 5e3
    prevtext = ""
    limit = string(taskf) == "turn_and_move_to_x" && limit > 1e5 ? 1e5 : limit
    @showprogress 1 "$(taskf) ... " for i in 1:Int(limit)
        ins, mp = taskf("temp", 1)
        ind += 1

        instext = join(ins.text, " ")
        #info(instext)
        if instext in unmatched
            push!(matched, instext)
            delete!(unmatched, instext)
            patlim += patience
            patience = 0
        else
            if instext != prevtext
                patience += 1
            end
        end
        prevtext = instext
        if length(unmatched) == 0 || patience >= patlim
            break
        end
    end
    numm = length(matched)
    numu = length(unmatched)
    matchedcount = 0
    for t in matched; matchedcount += freq[t]; end;

    matchedcount_nonsingle = 0
    for t in matched; matchedcount_nonsingle += (freq[t] != 1 ? freq[t] : 0); end;

    total = 0;
    for t in keys(freq); total += (freq[t] != 1 ? freq[t] : 0); end;

    ratio = matchedcount / sum(values(freq))
    ratio2 = matchedcount_nonsingle / total

    info("Task: $taskf")
    info("Total number of generated instances: $ind")

    info("Num of matched: $numm ($matchedcount / $(matchedcount_nonsingle)) ")
    for t in matched; info(t); info("Freq: $(freq[t])"); end;
    info("")
    info("Num of unmatched: $numu ($(sum(values(freq)) - matchedcount) / $(total - matchedcount_nonsingle))")
    for t in unmatched; info(t); info("Freq: $(freq[t])"); end;
    info("")
    info("Coverage of $(taskf): $(100.0 * ratio)%")
    info("Uniq Coverage of $(taskf): $(100.0 * (numm / (numm+numu)))%")
    info("NonSingle Coverage of $(taskf): $(100.0 * ratio2)%")
    return ratio, ratio2
end

function main(args=ARGS)
    tasklist = [turn_to_x, move_to_x, lang_only,
        turn_to_x_and_move, move_to_x_and_turn, turn_to_x_move_to_y,
        move_to_x_turn_to_y, move_until, orient, describe, turn_move_to_x,
        move_turn_to_x, turn_and_move_to_x, turn_move_until,
        turn_to_x_move_until, move_until_turn, move_until_turn_to_x]

    s = ArgParseSettings()
    @add_arg_table s begin
        ("--tasks"; nargs='*'; default=map(string, tasklist))
        ("--lname"; default="coverage.log")
        ("--oname"; default="taskcov.csv")
    end
    o = parse_args(args, s)
    Logging.configure(filename=o["lname"])
    Logging.configure(level=INFO)

    srand(123456789)

    grid, jelly, l = getallinstructions(;fname="../data/pickles/databag3.jld")
    alltext = map(ins->join(ins.text, " "), vcat(grid, jelly, l))
    allins = Set(alltext)
    freq = Dict{String, Int}()
    for t in alltext
        if t in keys(freq)
            freq[t] = freq[t] + 1
        else
            freq[t] = 1
        end
    end

    df = DataFrame(TaskName=Any[], Coverage=Any[])

    tasklist = map(x->eval(parse(x)), o["tasks"])

    total = 0.0
    total2 = 0.0
    for taskf in tasklist
        ratio, ratio2 = coverage(taskf, copy(allins), freq)
        push!(df, (taskf, ratio*100))
        total += ratio
        total2 = ratio2
    end

    info("Total coverage: $(total * 100)%")
    info("Total coverage (non single): $(total2 * 100)%")
    writetable(o["oname"], df)
end

main()
