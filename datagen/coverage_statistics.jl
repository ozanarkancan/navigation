using Logging, DataFrames, ProgressMeter, ArgParse

include("data_generator.jl")

function coverage(taskf, unmatched, freq, cond; limit=1e6)
    matched = Set()
    patience = 0
    ind = 0
    patlim = string(taskf) == "lang_only" ? 5e4 : 3e4
    prevtext = ""
    limit = string(taskf) == "turn_and_move_to_x" && limit > 1e5 ? 1e5 : limit

    @showprogress 1 "$(taskf) ... " for i in 1:Int(limit)
        ins, mp = taskf("temp", 1)
        ind += 1

        instext = join(ins.text, " ")
        info(instext)
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

    matchedcount = 0
    matchedcount_nonuniq = 0

    total = 0
    total_nonuniq = 0

    for t in keys(freq)
        for cat in keys(freq[t])
            if cat in cond
                if freq[t][cat] > 1
                    total_nonuniq += freq[t][cat]
                    if t in matched
                        matchedcount_nonuniq += freq[t][cat]
                    end
                end
                total += freq[t][cat]
                if t in matched
                    matchedcount += freq[t][cat]
                    info(string(taskf, " Matched: ", cat, " ", freq[t][cat], " ", t))
                else
                    info(string(taskf, " Not Matched: ", cat, " ", freq[t][cat], " ", t))
                end
            else
                info(string(taskf, " Not Matched: ", cat, " ", freq[t][cat], " ", t))
            end
        end
    end

    ratio = matchedcount / total
    ratio2 = matchedcount_nonuniq / total_nonuniq

    info("Task: $taskf")
    info("Total number of generated instances: $ind")

    info("Num of matched: $matchedcount / $(matchedcount_nonuniq)")
    info("Coverage of $(taskf): $(100.0 * ratio)%")
    info("NonUniq Coverage of $(taskf): $(100.0 * ratio2)%")
    return ratio, ratio2
end

function main(args=ARGS)
    tasklist = [(lang_only, ["Turn", "Move", "CombinationLang"]), (turn_to_x, ["TurnToX"]),
        (move_to_x,["MoveToX"]), (orient, ["Orient"]), (describe, ["Describe"]),
        (move_until, ["Conditional"]),(turn_and_move_to_x, ["TurnMoveToX"]), (any_combination, ["AnyCombination"])]

    s = ArgParseSettings()
    @add_arg_table s begin
        ("--lname"; default="coverage.log")
        ("--oname"; default="taskcov.csv")
    end
    o = parse_args(args, s)
    Logging.configure(filename=o["lname"])
    Logging.configure(level=INFO)

    srand(123456789)

    data = readtable("datacategory.csv")

    allins = Set(data[4])
    freq = Dict{String, Dict{String, Int}}()
    for i=1:size(data, 1)
        t = data[i, 4]
        cat = data[i, 5]
        if haskey(freq, t)
            if haskey(freq[t], cat)
                freq[t][cat] += 1
            else
                freq[t][cat] = 1
            end
        else
            freq[t] = Dict{String, Int}(cat=>1)
        end
    end

    df = DataFrame(TaskName=Any[], Coverage=Any[], NonUniqCoverage=Any[])

    for (taskf, cond) in tasklist
        ratio, ratio2 = coverage(taskf, copy(allins), freq, cond)
        push!(df, (taskf, ratio*100, ratio2*100))
    end

    writetable(o["oname"], df)
end

main()
