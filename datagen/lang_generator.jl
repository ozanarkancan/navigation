include("navimap_utils.jl")

@enum Category visual_t visual_m visual_tm langonly_t langonly_m langonly_s orient_t condition_m description empty

opposites = Dict(0=>"south", 90=>"west", 180=>"north", 270=>"east")
rights = Dict(0=>"east", 90=>"south", 180=>"west", 270=>"north")
lefts = Dict(0=>"west", 90=>"north", 180=>"east", 270=>"south")
ordinals = Dict(1=>"first", 2=>"second", 3=>"third", 4=>"fourth", 5=>"fifth", 
    6=>"sixth", 7=>"seventh", 8=>"eighth", 9=>"ninth")
times = Dict(1=>"once", 2=>"twice")
numbers = Dict(1=>["one", "a"],2=>["two", "2"],3=>["three", "3"],4=>["four", "4"],5=>["five", "5"],
    6=>["six", "6"],7=>["seven", "7"],8=>["eight", "8"],9=>["nine", "9"],10=>["ten", "10"])
wall_names = Dict(1=>"butterflies",2=>"fish",3=>"towers")
floor_names = Dict(1=>["octagon", "blue-tiled"],2=>["brick"],3=>["bare concrete", "concrete", "plain cement"],
    4=>["flower", "flowered", "pink-flowered", "rose"], 5=>["grass", "grassy"],6=>["gravel", "stone"],
    7=>["wood", "wooden", "wooden-floored"],8=>["yellow", "yellow-tiled", "honeycomb yellow"])
item_names = Dict(1=>["stool"], 2=>["chair"], 3=>["easel"], 4=>["hatrack", "hat rack", "coatrack", "coat rack"],
    5=>["lamp"], 6=>["sofa", "bench"])

function action(curr, next)
    a = 0
    if curr[1] != next[1] || curr[2] != next[2]#move
        a = 1
    elseif !(next[3] == 270 && curr[3] == 0) && (next[3] > curr[3] || (next[3] == 0 && curr[3] == 270))#right
        a = 2
    elseif !(next[3] == 0 && curr[3] == 270) && (next[3] < curr[3] || (next[3] == 270 && curr[3] == 0))#left
        a = 3
    else
        a = 4
    end
    return a
end				

function generate_lang(navimap, maze, segments; combine=0.6, cons=[])
    generation = Any[]

    if length(segments) > 1
        append!(generation, startins(navimap, maze, segments[1], segments[2]; cons=cons))
    end

    ind = 2
    while ind < length(segments)
        if rand() >= combine || ind+2 >= length(segments)
            g = (segments[ind][1] == "turn" ? turnins : moveins)(navimap, maze, segments[ind], segments[ind+1]; cons=cons)
            ind += 1
            if rand() <= 0.05
                g[1] = (g[1][1], (string("then ", g[1][2][1]), g[1][2][2:end]...))
            end
        else
            g = (segments[ind][1] == "turn" ? turnmoveins : moveturnins)(navimap, maze, segments[ind], segments[ind+1], segments[ind+2]; cons=cons)
            ind += 2
        end

        append!(generation, g)
    end
    append!(generation, finalins(navimap, maze, segments[end]; cons=cons))
    return generation
end

function to_string(generation)
    txt = ""
    for (s, ins) in generation
        txt = string(txt, "\n", s, "\n", ins, "\n")
    end
    return txt
end

function startins(navimap, maze, curr, next; cons=[])
    """
    TODO
    """
    curr_t, curr_s = curr
    next_t, next_s = next

    a = action(curr_s[1], curr_s[2])
    p1 = (curr_s[1][2], curr_s[1][1], -1)

    if curr_t == "turn"
        cands = Any[]
        dir = ""
        d = ""
        
        if (length(cons) == 0 || langonly_t in cons)
            if length(curr_s) == 2
                if a == 2#right
                    dir = rights[curr_s[1][3]]
                    d = "right"
                else#left
                    dir = lefts[curr_s[1][3]]
                    d = "left"
                end
                push!(cands, (string("turn ", d), langonly_t))
            else
                push!(cands, ("turn around", langonly_t))
                dir = opposites[curr_s[1][3]]
            end
        end

        diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

        if diff_w && (length(cons) == 0 || visual_t in cons)
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                for cor in ["corridor ", "hall ", "alley ", "hallway "]
                    for sufx in ["", " on the wall", " on both sides of the walls"]
                        for with in ["with the ", "with the pictures of "]
                            push!(cands, (string(prefx, cor, with, wall_names[wpatrn], sufx), visual_t))
                        end
                    end
                end
            end
        end

        if diff_f && (length(cons) == 0 || visual_t in cons)
            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                    if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, (string(prefx, flr, cor), visual_t))
                    end
                end
            end

            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                    push!(cors, " carpet")
                end
                for cor in cors
                    for verb in ["facing the ", "seeing the "]
                        push!(cands, (string("you should be ", verb, flr, cor), visual_t))
                    end
                end
            end
        end

        if d != "" && diff_f && (length(cons) == 0 || visual_t in cons)
            for prefx in [""]
                for cor in [" corridor", " hall", " alley", " hallway", " path", "", " floor", "flooring"]
                    for v in ["take a $d into ", "make a $d into ", "take a $d onto ", "make a $d onto ","turn $d into ", "turn $d onto ", "turn $d to face "]
                        for det in ["the ", ""]
                            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                                push!(cands, (string(prefx, v, det, flr, cor), visual_t))
                            end
                        end
                    end
                end
            end
        end

        len = nothing
        if length(curr_s) == 2 && is_intersection(maze, p1) && (length(cons) == 0 || visual_t in cons)
            left_hall = hall_front(navimap, getlocation(navimap, curr_s[1], 3))
            right_hall = hall_front(navimap, getlocation(navimap, curr_s[1], 2))
            l_l = length(left_hall)
            l_r = length(right_hall)

            if l_l != l_r
                if a == 2
                    len = l_r > l_l ? ["long", "longer"] : ["short", "shorter"]
                else
                    len = l_l > l_r ? ["long", "longer"] : ["short", "shorter"]
                end

                for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]

                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for l in len
                                push!(cands, (string(prefx, l, " ", flr, cor), visual_t))
                            end
                        end
                    end
                    for cor in cors
                        for l in len
                            push!(cands, (string(prefx, l, " ", cor), visual_t))
                        end
                    end
                end
            end
        end
        
        item = find_single_item_in_visible(navimap, curr_s[1][1:2], next_s[2])
        if item != 7 && (length(cons) == 0 || visual_t in cons)
            for prefx in [""]
                for body in ["look for the ", "face the ", "face toward the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                    if body == "look for the "
                        push!(cands, (string(prefx, body, rand(item_names[item])), visual_t))
                    else
                        det = item == 3 ? "an" : "a"
                        for cor in ["corridor ", "hall ", "alley ", "hallway ", "path ", "intersection ", "segment ", ""]
                            sfxs = cor == "" ? [""] : ["with $det ", "containing the ", "with the "]
                            for suffx in sfxs
                                push!(cands, (string(prefx, body, cor, suffx, rand(item_names[item])), visual_t))
                            end
                        end
                        
                        for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                            cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                            if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                                push!(cors, " carpet")
                            end
                            for cor in cors
                                sfxs = cor == "" ? [""] : [" with $det ", " containing the ", " with the "]
                                for suffx in sfxs
                                    push!(cands, (string(prefx, body, flr, cor, suffx, rand(item_names[item])), visual_t))
                                end
                                push!(cands, (string(prefx, body, rand(item_names[item]), " in the ", flr, cor), visual_t))
                            end
                        end

                    end
                end
            end

            for v in ["turn ", "go ", "turn to the ", "make a ", "take a "]
                for to in [" to the ", " toward the ", " towards the "]
                    push!(cands, (string(v, d, to, rand(item_names[item])), visual_t))
                end
            end
        end

        if is_deadend(maze, p1) && (length(cons) == 0 || visual_t in cons)
            push!(cands, ("you should leave the dead end", visual_t))
            for w in ["way ", "direction "]
                for g in ["go", "move", "travel", "walk"]
                    push!(cands, (string("only one ", w, "to ", g), visual_t))
                end
            end
        end

        if sum(maze[p1[1], p1[2], :]) == 3 || sum(maze[p1[1], p1[2], :]) == 2 && (length(cons) == 0 || orient_t in cons)
            p = (curr_s[end][2], curr_s[end][1], round(Int, 1+curr_s[end][3] / 90))
            rightwall = maze[p[1], p[2], rightof(p[3])] == 0
            leftwall = maze[p[1], p[2], leftof(p[3])] == 0
            backwall = maze[p[1], p[2], backof(p[3])] == 0

            if rightwall && !backwall && !leftwall
                push!(cands, ("turn so that the wall is on your right", orient_t))
                push!(cands, ("turn so that the wall is on your right side", orient_t))
                push!(cands, ("turn so the wall is on your right", orient_t))
                push!(cands, ("turn so the wall is on your right side", orient_t))
            elseif rightwall && backwall && !leftwall
                push!(cands, ("turn so that the wall is on your right and back", orient_t))
                push!(cands, ("turn so that the wall is on your back and right", orient_t))
                push!(cands, ("turn so the wall is on your back and right", orient_t))
                push!(cands, ("turn so the wall is on your right and right", orient_t))
            elseif !rightwall && !backwall && leftwall
                push!(cands, ("turn so that the wall is on your left", orient_t))
                push!(cands, ("turn so that the wall is on your left side", orient_t))
                push!(cands, ("turn so the wall is on your left", orient_t))
                push!(cands, ("turn so the wall is on your left side", orient_t))
            elseif !rightwall && backwall && leftwall
                push!(cands, ("turn so that the wall is on your left and back", orient_t))
                push!(cands, ("turn so the wall is on your left and back", orient_t))
                push!(cands, ("turn so that the wall is on your back and left", orient_t))
                push!(cands, ("turn so the wall is on your back and left", orient_t))
            elseif !rightwall && backwall && !leftwall
                push!(cands, ("turn so that the wall is on your back", orient_t))
                push!(cands, ("turn so that the wall is to your back", orient_t))
                push!(cands, ("turn so the wall is on your back", orient_t))
                push!(cands, ("turn so the wall is to your back", orient_t))
                push!(cands, ("turn so that the wall is on your back side", orient_t))
                push!(cands, ("turn so the wall is on your back side", orient_t))
                push!(cands, ("turn so that your back is to the wall", orient_t))
                push!(cands, ("turn so that your back faces the wall", orient_t))
                push!(cands, ("turn so that your back side faces the wall", orient_t))
                push!(cands, ("stand with your back to the wall of the 't' intersection", orient_t))
                push!(cands, ("place your back to the 't' intersection", orient_t))
                for r in [" to", " against"]
                    for suffix in [" of the 't' intersection", ""]
                        push!(cands, (string("place your back", r, " the wall", suffix), orient_t))
                    end
                end
            end
            
            if length(curr_s) == 2 && sum(maze[p1[1], p1[2], :]) == 3 
                d = a == 2 ? "right" : "left"
                p = (curr_s[1][2], curr_s[1][1], round(Int, 1+curr_s[1][3] / 90))
                rightwall = maze[p[1], p[2], rightof(p[3])] == 0
                leftwall = maze[p[1], p[2], leftof(p[3])] == 0
                backwall = maze[p[1], p[2], backof(p[3])] == 0
            
                if !rightwall && backwall && !leftwall
                    push!(cands, (string("with your back to the wall turn ", d), orient_t))
                end
            end

        end

        if length(cands) != 0
            return  [(curr_s, rand(cands))]
        else
            return [(curr_s, ("", empty))]
        end
    else
        #diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
        #wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        l = Any[]

        append!(l, moveins(navimap, maze, curr, next; cons=cons))
        return l
    end
end

function moveins(navimap, maze, curr, next; cons=[])
    curr_t, curr_s = curr
    next_t = next != nothing ? next[1] : nothing
    next_s = next != nothing ? next[2] : nothing

    endpoint = map(x->round(Int, x), curr_s[end])
    d = round(Int, endpoint[3] / 90 + 1)

    cands = Any[]
    steps = length(curr_s)-1

    sts = steps > 1 ? [" steps", " blocks", " segments", " times"] : [" step", " block", " segment", " space"]
    if (length(cons) == 0 || langonly_m in cons)
        for g in ["go ", "move ", "walk "]
            for m in ["forward ", "straight ", ""]
                if steps < 3
                    push!(cands, (string(g, m, times[steps]), langonly_m))
                end
                for st in sts
                    for num in numbers[steps]
                        push!(cands, (string(g, m, num, st), langonly_m))
                    end
                end
            end
        end

        for v in ["take ", ""]
            for num in numbers[steps]
                for st in sts
                    push!(cands, (string(v, num, st), langonly_m))
                end
            end
        end
    end

    if facing_wall(maze, (endpoint[2], endpoint[1], d)) && (length(cons) == 0 || visual_m in cons)
        for cor in [" path", " hall", " hallway", " alley", " corridor"]
            for unt in [" until ", " until you get to ", " until you reach "]
                push!(cands, (string("take the", cor, unt, "the wall"), visual_m))
            end
        end

        for m in ["move ", "go ", "walk "]
            for adv in ["forwards ", "straight ", ""]
                push!(cands, (string(m, adv, "as far as you can"), visual_m))
                for unt in ["until ", "until you get to ", " until you reach "]
                    push!(cands, (string(m, adv, unt, "the wall"), visual_m))
                end
            end
        end
    end

    p1 = (curr_s[1][2], curr_s[1][1], -1)
    p2 = (curr_s[end][2], curr_s[end][1], -1)

    if is_corner(maze, p2) && (length(cons) == 0 || visual_m in cons)
        for cor in [" path", " hall", " hallway", " alley", " corridor"]
            for prep in [" into the corner", " to the corner", " to the next corner"]
                push!(cands, (string("follow the ", cor, prep), visual_m))
            end
        end

        for m in ["move ", "go ", "walk "]
            for adv in [" forward", " straight", ""]
                for prep in [" into the corner", " to the corner", " to the next corner"]
                    for suffix in ["", " you see in front of you"]
                        push!(cands, (string(m, adv, prep, suffix), visual_m))
                        for st in sts
                            for num in numbers[steps]
                                push!(cands, (string(m, adv, " ", num, st, prep, suffix), visual_m))
                            end
                        end
                    end
                end
            end
        end
    end

    if is_deadend(maze, p2) && (length(cons) == 0 || visual_m in cons)
        for m in ["move ", "go ", "walk "]
            for adv in ["forward", "straight", ""]
                push!(cands, (string(m, adv, " into the dead end"), visual_m))
            end
        end
    end

    if (is_corner(maze, p1) || is_deadend(maze, p1)) && (is_corner(maze, p2) || is_deadend(maze, p2)) && (length(cons) == 0 || visual_m in cons)
        for m in ["move ", "go ", "walk "]
            for adv in ["forward ", "straight ", ""]
                for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                    if sufx != ""
                        push!(cands, (string(m, adv, "to the other end of the ", sufx), visual_m))
                    else
                        push!(cands, (string(m, adv, "to the other end"), visual_m))
                    end
                    for st in sts
                        for num in numbers[steps]
                            if sufx != ""
                                push!(cands, (string(m, adv, num, st, " to the other end of the ", sufx), visual_m))
                            else
                                push!(cands, (string(m, adv, num, st, " to the other end"), visual_m))
                            end
                        end
                    end
                end
            end
        end
    elseif (is_corner(maze, p2) || is_deadend(maze, p2)) && (length(cons) == 0 || visual_m in cons)
        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        for m in ["move ", "go ", "walk "]
            for adv in ["forward ", "straight ", ""]
                for sufx in ["hall", "hallway", "path", "corridor", "alley", ""]
                    if sufx != ""
                        push!(cands, (string(m, adv, "all the way to the end of the ", sufx), visual_m))
                        push!(cands, (string(m, adv, "to the end of the ", sufx), visual_m))
                    else
                        push!(cands, (string(m, adv, "to the end"), visual_m))
                    end
                    
                    for st in sts
                        for num in numbers[steps]
                            if sufx != ""
                                push!(cands, (string(m, adv, num, st, " all the way to the end of the ", sufx), visual_m))
                                push!(cands, (string(m, adv, num, st, " to the end of the ", sufx), visual_m))
                            else
                                push!(cands, (string(m, adv, num, st, " to the end"), visual_m))
                            end
                        end
                    end
                end
                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                    if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        push!(cands, (string(m, adv, "to the end of the ", flr, cor), visual_m))
                        for st in sts
                            for num in numbers[steps]
                                push!(cands, (string(m, adv, num, st, " to the end of the ", flr, cor), visual_m))
                                push!(cands, (string(m, adv, "to the end of the ", flr, cor, " ", num, st), visual_m))
                            end
                        end
                    end
                end
            end
        end
    elseif is_deadend(maze, p2) && (length(cons) == 0 || visual_m in cons)
        for m in ["move ", "go ", "walk "]
            for adv in ["forward ", "straight ", ""]
                push!(cands, (string(m, adv, "to the deadend"), visual_m))
                push!(cands, (string(m, adv, "to the dead end"), visual_m))
                push!(cands, (string(m, adv, "until you get to a deadend"), visual_m))
                push!(cands, (string(m, adv, "until you get to a dead end"), visual_m))
                    
                for st in sts
                    for num in numbers[steps]
                        push!(cands, (string(m, adv, num, st, " to the deadend"), visual_m))
                        push!(cands, (string(m, adv, num, st, " to the dead end"), visual_m))
                    end
                end
            end
        end
    end

    if is_intersection(maze, p2) && (length(cons) == 0 || condition_m in cons)
        alleycnt = count_alleys(maze, curr_s)
        if alleycnt > 0
            for m in ["move", "go", "walk"]
                for cond in [" until the ", " to the "]
                    if alleycnt == 1
                        push!(cands, (string(m, cond, "next alley"), condition_m))
                    else
                        push!(cands, (string(m, cond, ordinals[alleycnt], " alley"), condition_m))
                    end
                end
            end
        end
    end

    if navimap.nodes[curr_s[end][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[curr_s[end][1:2]], curr_s[1][1:2]) == 1 && (length(cons) == 0 || visual_m in cons)
        det = navimap.nodes[curr_s[end][1:2]] == 3 ? "an" : "a"
        for m in ["go ", "move ", "walk "]
            for adv in ["forward ", "straight ", "", "on the path "]
                for num in numbers[steps]
                    for st in sts
                        for tow in [" to", " towards", " toward"]
                            for suffix in [" the intersection containing the ", " the intersection with $det ", " the intersection has $det "]
                                push!(cands, (string(m, adv, num, st, tow, suffix,
                                    rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                            end
                        end
                    end
                end
            end
        end

        for v in ["take ", "", "follow this path ", "follow this hall ", "follow this hallway "]
            for num in numbers[steps]
                for st in sts
                    for tow in [" to", " towards", " toward"]
                        push!(cands, (string(v, num, st, tow, " the intersection containing the ",
                            rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                    end
                end
            end
        end

        for cond in ["until the ", "toward the ", "towards the ", "until you get to ", "until yo get $det ", "until you get to the ", "until you reach the ", "till you get to $det ", "to the "]

            for m in ["go ", "move ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    push!(cands, (string(m, adv, cond, rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                end
            end
        end

        for cor in ["path", "hall", "hallway"]
            for v in ["take the ", "move along the ", "go along the ", "walk along the "]
                push!(cands, (string(v, cor, rand([" towards the ", " toward the ", " to the "]), rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
            end
        end

        wpatrn, fpatrn = navimap.edges[(curr_s[1][1], curr_s[1][2])][(curr_s[2][1], curr_s[2][2])]
        for v in ["follow the ", "along the ", "take the ", "follow this "]
            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                    push!(cors, " carpet")
                end
                for cor in cors
                    push!(cands, (string(v, flr, cor, " to the ", rand(item_names[navimap.nodes[curr_s[end][1:2]]])), visual_m))
                end
            end
        end
    elseif navimap.nodes[curr_s[end][1:2]] == 7 && navimap.nodes[curr_s[end-1][1:2]] != 7 && 
        length(curr_s) > 2 && item_single_in_visible(navimap, navimap.nodes[curr_s[end-1][1:2]], curr_s[1][1:2]) == 1 &&
        (length(cons) == 0 || condition_m in cons)
        for m in ["move ", "go ", "walk "]
            for one in ["a", "one"]
                for step in [" step", " block", " segment"]
                    push!(cands, (string(m, one, step, " beyond the ",
                        rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))
                end
            end
        end

        det = navimap.nodes[curr_s[end-1][1:2]] == 3 ? "an" : "a"
        for prefx in ["one block pass the ", "pass the ", "move past the ", "you will pass $det "]
            push!(cands, (string(prefx, rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))
        end

        for num in numbers[steps]
            for st in sts
                for body in [" past ", " passing ", " passing the ", " passing $det "]
                    push!(cands, (string(num, st, body,
                        rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))

                    for m in ["go ", "move ", "walk "]
                        for adv in ["forward ", "straight ", ""]
                            push!(cands, (string(m, adv, num, st, body,
                                rand(item_names[navimap.nodes[curr_s[end-1][1:2]]])), condition_m))
                        end
                    end
                end
            end
        end
    end

    if steps < 3 && next != nothing && (length(cons) == 0 || visual_m in cons)
        target = getlocation(navimap, next_s[2], 1)
        res, fpatrn = is_floor_unique(navimap, maze, curr_s, target)
        if res != 0
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for cond in ["to the intersection with the ", "to the "]
                                for st in sts
                                    for num in numbers[steps]
                                        push!(cands, (string(m, adv, num, st, cond, flr, cor), condition_m))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if steps >= 3 && next != nothing && (length(cons) == 0 || condition_m in cons)
        wp, fp = navimap.edges[curr_s[1][1:2]][curr_s[2][1:2]]

        target = getlocation(navimap, next_s[2], 1)

        res, fpatrn = is_floor_unique(navimap, maze, curr_s, target)
        if res != 0
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", "", "to the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            push!(cands, (string(m, adv, flr, cor), condition_m))
                        end
                    end
                end
            end
        end

        if res == 1
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" on your right"]
                                push!(cands, (string(m, adv, "until you see the ",
                                    flr, cor, d), condition_m))
                            end
                        end
                    end
                end
            end
        elseif res == 2
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for d in [" on your left"]
                                push!(cands, (string(m, adv, "until you see the ",
                                    flr, cor, d), condition_m))
                            end
                        end
                    end
                end
            end
        elseif res == 3
            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            for cond in ["until you reach the ", "to the intersection with the ", "to the "]
                                push!(cands, (string(m, adv, cond, flr, cor), condition_m))
                            end
                        end
                    end
                end
            end

            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for c in ColorMapping[fpatrn]
                        push!(cands, (string(m, adv, "until you reach the ", c, " intersection"), condition_m))
                    end
                end
            end

            for m in ["move ", "go ", "walk "]
                for adv in ["forward ", "straight ", ""]
                    for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                        for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                            push!(cands, (string(m, adv, "until you reach an intersection with ",
                            flr1, " and ", flr2), condition_m))
                        end
                    end
                end
            end

            for v in ["take the ", "follow the ", "follow this ", "move along the ", "walk along the ", "go along the ", "go forward along the "]
                for flr1 in vcat(floor_names[fp], ColorMapping[fp])
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        for cor1 in [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                            for cond in [" to the intersection with the ", " until it crosses the ", " until you end up on the ", " to the "]
                                for cor2 in [" path", " hall", " hallway", " alley", " corridor", " floor", " flooring"]
                                    push!(cands, (string(v, flr1, cor1, cond, flr2, cor2), condition_m))
                                end
                            end
                        end
                    end
                end
            end

            for cor1 in [" path", " hall", " hallway", " alley", " corridor"]
                for cond in [" until you reach the ", " end up on the "]
                    for flr2 in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        for cor2 in [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                            push!(cands, (string("follow this", cor1, cond, flr2, cor2), condition_m))
                        end
                    end
                end
            end
        end
    end

    if length(cands) != 0
        return [(curr_s, rand(cands))]
    else
        return [(curr_s, ("", empty))]
    end
end

"""
TODO
"""
function turnins(navimap, maze, curr, next; cons=[])
    curr_t, curr_s = curr
    next_t, next_s = next

    cands = Any[]
    a = action(curr_s[1], curr_s[2])
    d = a == 2 ? "right" : "left"

    if (length(cons) == 0 || langonly_t in cons)
        for v in ["turn ", "go ", "turn to the ", "make a ", "take a "]
            push!(cands, (string(v, d), langonly_t))
        end

        if is_corner(maze, (curr_s[1][2], curr_s[1][1], round(Int, curr_s[1][3]/90 + 1)))
            push!(cands, (string("at the corner turn ", d), langonly_t))
            push!(cands, (string("turn ", d, " at the corner"), langonly_t))
        end
    end

    diff_w, diff_f = around_different_walls_floor(navimap, (curr_s[1][1], curr_s[1][2]))
    wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]

    if diff_w && (length(cons) == 0 || visual_t in cons)
        for prefx in [""]
            for cor in ["corridor ", "hall ", "alley ", "hallway ", "path "]
                for v in ["look for the ", "face the ", "turn your face to the ", "turn to the ", "turn until you see the "]
                    for sufx in ["", " on the wall", " on both sides of the walls"]
                        for with in ["with the ", "with the pictures of "]
                            push!(cands, (string(prefx, cor, with, wall_names[wpatrn], sufx), visual_t))
                        end
                    end
                end
            end
        end
    end

    if diff_f && (length(cons) == 0 || visual_t in cons)
        for prefx in [""]
            for cor in [" corridor", " hall", " alley", " hallway", " path", "", " floor", " flooring"]
                for v in ["look for the ", "face the ", "turn your face to the ", "turn to the ", "turn until you see the ", "turn onto the ", "turn into the ", "turn to face the ", "orient yourself to face the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, (string(prefx, v, flr, cor), visual_t))
                    end
                end
            end
        end

        for prefx in [""]
            for cor in [" corridor", " hall", " alley", " hallway", " path", "", " floor", " flooring"]
                for v in ["facing the ", "seeing the "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        push!(cands, (string(prefx, "you should be ", v, flr, cor), visual_t))
                    end
                end
            end
        end
    end

    if diff_f && (length(cons) == 0 || visual_t in cons)
        for prefx in [""]
            for cor in [" corridor", " hall", " alley", " hallway", " path", "", " floor", " flooring"]
                for v in ["take a $d into ", "make a $d into ", "take a $d onto ", "make a $d onto ","turn $d into ", "turn $d onto ", "turn $d to face "]
                    for det in ["the ", ""]
                        for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                            push!(cands, (string(prefx, v, det, flr, cor), visual_t))
                        end
                    end
                end
            end
        end
    end
    
    p1 = (curr_s[1][2], curr_s[1][1], -1)
    len = nothing
    if length(curr_s) == 2 && is_intersection(maze, p1) && (length(cons) == 0 || visual_t in cons)
        left_hall = hall_front(navimap, getlocation(navimap, curr_s[1], 3))
        right_hall = hall_front(navimap, getlocation(navimap, curr_s[1], 2))
        l_l = length(left_hall)
        l_r = length(right_hall)

        if l_l != l_r
            if a == 2
                len = l_r > l_l ? ["long", "longer"] : ["short", "shorter"]
            else
                len = l_l > l_r ? ["long", "longer"] : ["short", "shorter"]
            end

            for prefx in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]

                for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                    cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                    if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                        push!(cors, " carpet")
                    end
                    for cor in cors
                        for l in len
                            push!(cands, (string(prefx, l, " ", flr, cor), visual_t))
                        end
                    end
                end
                for cor in cors
                    for l in len
                        push!(cands, (string(prefx, l, " ", cor), visual_t))
                    end
                end
            end
        end
    end

    item = find_single_item_in_visible(navimap, curr_s[1][1:2], next_s[2])
    if item != 7 && (length(cons) == 0 || visual_t in cons)
        for prefx in [""]
            for body in ["look for the ", "face the ", "face toward the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                if body == "look for the "
                    push!(cands, (string(prefx, body, rand(item_names[item])), visual_t))
                else
                    det = item == 3 ? "an" : "a"
                    for cor in ["corridor ", "hall ", "alley ", "hallway ", "path ", "intersection ", "segment ", ""]
                        sfxs = cor == "" ? [""] : ["with $det ", "containing the ", "with the "]
                        for suffx in sfxs
                            push!(cands, (string(prefx, body, cor, suffx, rand(item_names[item])), visual_t))
                        end
                    end

                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            sfxs = [" with $det ", " containing the ", " with the "]
                            for suffx in sfxs
                                push!(cands, (string(prefx, body, flr, cor, suffx, rand(item_names[item])), visual_t))
                            end
                            push!(cands, (string(prefx, body, rand(item_names[item]), " in the ", flr, cor), visual_t))
                        end
                    end

                end
            end
        end
        
        for v in ["turn ", "go ", "turn to the ", "make a ", "take a "]
            for to in [" to the ", " toward the ", " towards the "]
                push!(cands, (string(v, d, to, rand(item_names[item])), visual_t))
            end
        end
        
        for prefx in [""]
            for body in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                push!(cands, (string(prefx, body, rand(item_names[item])), visual_t))
            end
        end
    end

    if navimap.nodes[curr_s[1][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[curr_s[1][1:2]], curr_s[1][1:2]) == 0 &&
        (length(cons) == 0 || langonly_t in cons)
        prefx = "at the "
        for suffix in [" turn ", " take a ", " turn to the ", " make a ", " go ", " move "]
            push!(cands, (string(prefx, rand(item_names[navimap.nodes[curr_s[1][1:2]]]), suffix, d), langonly_t))
        end
    end
    
    if length(cands) != 0
        return [(curr_s, rand(cands))]
    else
        return [(curr_s, ("", empty))]
    end
end

function moveturnins(navimap, maze, curr, next, next2; cons=[])
    steps = length(curr)-1
    cands = Any[]
    
    curr_t, curr_s = curr
    next_t, next_s = next

    segm = copy(curr_s)
    append!(segm, next_s[2:end])


    p1 = (curr_s[1][2], curr_s[1][1], -1)
    p2 = (curr_s[end][2], curr_s[end][1], -1)
   
    a = action(next_s[1], next_s[2])
    d = a == 2 ? "right" : "left"

    if is_intersection(maze, p2) && (length(cons) == 0 || (langonly_t in cons && condition_m in cons))
        alleycnt = count_alleys(maze, curr_s)
        if alleycnt == 1
            push!(cands, (string("make the first ", d), condition_m, langonly_t))
        end
    end

    if is_corner(maze, p2) && (length(cons) == 0 || (visual_m in cons && langonly_t in cons))
        for cond in ["at the end ", "at the end of this hall ", "at the end of the hall ", "when the hall ends ", "at the corner ", "at the next corner "]
            for suffix in ["turn ", "take a ", "turn to the ", "make a ", "go ", "move "]
                push!(cands, (string(cond, suffix, d), visual_m, langonly_t))
            end
        end
    end
    
    item = find_single_item_in_visible(navimap, curr_s[end][1:2], next_s[2])
    if item != 7 && is_corner(maze, p2) && (length(cons) == 0 || (visual_m in cons && visual_t in cons))
        for prefx in ["at the end ", "at the end of this hall ", "at the end of the hall ", "when the hall ends ", "at the corner "]
            for body in ["look for the ", "face the ", "turn your face to the ", "turn until you see the ", "turn to the ", "turn to face the ", "orient yourself to face the "]
                push!(cands, (string(prefx, body, rand(item_names[item])), visual_m, visual_t))
            end
        end
    end

    if navimap.nodes[curr_s[end][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[curr_s[end][1:2]], curr_s[1][1:2]) == 1 && (length(cons) == 0 || (visual_m in cons && visual_t in cons))
        item = navimap.nodes[curr_s[end][1:2]]
        for suffix in [" turn ", " take a ", " turn to the ", " make a ", " go ", " move "]
            push!(cands, (string("at the ", rand(item_names[item]), suffix, d), visual_m, langonly_t))
        end
    end

    if length(cands) == 0 || rand() < 0.5
        mins = moveins(navimap, maze, curr, next; cons=cons)
        tins = turnins(navimap, maze, next, next2; cons=cons)

        ts, ti = tins[1]
        ms, mi = mins[1]

        append!(ms, ts[2:end])
        newins = string(mi[1], rand([" and ", " then ", " and then ", " "]), ti[1])
        return [(ms, (newins, mi[2], ti[2]))]
    end
    if length(cands) != 0
        return [(segm, rand(cands))]
    else
        return [(segm, ("", empty))]
    end
end

function turnmoveins(navimap, maze, curr, next, next2; cons=[])
    steps = length(next)-1
    cands = Any[]
    curr_t, curr_s = curr
    next_t, next_s = next

    segm = copy(curr_s)
    append!(segm, next_s[2:end])

    steps = length(next_s)-1
    sts = steps > 1 ? [" steps", " blocks", " segments", " times"] : [" step", " block", " segment"]

    if navimap.nodes[next_s[end][1:2]] != 7 && item_single_in_visible(navimap, navimap.nodes[next_s[end][1:2]], curr_s[1][1:2]) == 1 &&
        (length(cons) == 0 || visual_tm in cons)
        for tv in ["turn and ", "face and "]
            for mv in ["move ", "go ", "walk "]
                for m in ["forward ", "straight ", ""]
                    push!(cands, (string(tv, mv, m,
                        rand(["to the ", "towards the ", "toward the "]), rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
                end
            end
        end
        for v in ["move ", "go ", "walk "]
            for to in ["towards the ", "toward the ", "to the "]
                push!(cands, (string(v, to, rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
            end
        end

        push!(cands, (string("take the ", rand(["path", "hall"])," towards the ", rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))

        wpatrn, fpatrn = navimap.edges[(next_s[1][1], next_s[1][2])][(next_s[2][1], next_s[2][2])]
        for v in ["turn and follow the ", "along the ", "face and follow the "]
            for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                for path in [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                    push!(cands, (string(v, flr, path, " to the ", 
                        rand(item_names[navimap.nodes[next_s[end][1:2]]])), visual_tm))
                end
            end
        end
    end

    if navimap.nodes[next_s[end][1:2]] == 7 && navimap.nodes[next_s[end-1][1:2]] != 7 &&
        length(next_s) > 2 && item_single_in_visible(navimap, navimap.nodes[next_s[end-1][1:2]], next_s[1][1:2]) == 1 &&
        (length(cons) == 0 || (langonly_t in cons && condition_m in cons))
        
        tprefx = "turn and "
        for m in ["move ", "go ", "walk "]
            for one in ["a", "one"]
                for step in [" step", " block", " segment"]
                    push!(cands, (string(tprefx, m, one, step, " beyond the ",
                        rand(item_names[navimap.nodes[next_s[end-1][1:2]]])), langonly_t, condition_m))
                end
            end
        end

        for prefx in ["one block pass the ", "pass the ", "move past the "]
            push!(cands, (string(tprefx, prefx, rand(item_names[navimap.nodes[next_s[end-1][1:2]]])), langonly_t, condition_m))
        end

        for num in numbers[steps]
            for st in sts
                for body in [" past ", " passing ", " passing the ", " passing a "]
                    push!(cands, (string(tprefx, num, st, body,
                        rand(item_names[navimap.nodes[next_s[end-1][1:2]]])), langonly_t, condition_m))

                    for m in ["go ", "move ", "walk "]
                        for adv in ["forward ", "straight ", ""]
                            push!(cands, (string(tprefx, m, adv, num, st, body,
                                rand(item_names[navimap.nodes[next_s[end-1][1:2]]])), langonly_t, condition_m))
                        end
                    end
                end
            end
        end
    end

    if length(cands) == 0 || rand() < 0.5
        tins = turnins(navimap, maze, curr, next; cons=cons)
        mins = moveins(navimap, maze, next, next2; cons=cons)

        ts, ti = tins[1]
        ms, mi = mins[1]

        append!(ts, ms[2:end])
        newins = string(ti[1], rand([" and ", " then ", " and then ", " "]), mi[1])
        return [(ts, (newins, ti[2], mi[2]))]
    end

    if length(cands) != 0
        return [(segm, rand(cands))]
    else
        return [(segm, ("", empty))]
    end
end

function finalins(navimap, maze, curr; cons=[])
    """
    TODO
    """
    curr_t, curr_s = curr
    cands = Any[]

    lasti = ""

    if curr_t == "turn"
        insl = turnins(navimap, maze, curr, nothing; cons=cons)
    else
        insl = moveins(navimap, maze, curr, nothing; cons=cons)
    end
    lasts, lasti = insl[end]

    r = rand()
    if r <= 0.4
        num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
        push!(cands, (string(lasti[1], " and that is the ", rand(["target ", "final "]), "position"),lasti[2]))
        push!(cands, (string(lasti[1], " and stop"),lasti[2]))
        push!(cands, (string(lasti[1], " then stop"),lasti[2]))
        push!(cands, (string(lasti[1], " and that is the position ", num), lasti[2]))
        push!(cands, (string(lasti[1], " and there should be the position ", num), lasti[2]))

        insl[end] = (lasts, rand(cands))
        return insl
    elseif r <= 0.8 && (length(cons) == 0 || description in cons || langonly_s in cons)
        if langonly_s in cons
            for v in ["that's it", "and stop", "then stop", "stop", "stop here"]
                push!(cands, (v, langonly_s))
            end
        else
            num = rand([rand(numbers[rand(2:10)]), rand(2:10)])
            push!(cands, (string("that is the ", rand(["target ", "final "]), "position"), description))
            push!(cands, (string("that is the position ", num), description))
            push!(cands, (string("there should be the position ", num), description))
            push!(cands, (string("position ", num, " should be there"), description))

            if navimap.nodes[curr_s[end][1:2]] != 7
                det = navimap.nodes[curr_s[end][1:2]] == 3 ? "an" : "a"
                for prefx in ["this intersection contains $det ", "there is $det ", "there should be $det ", "this contains $det "]
                    push!(cands, (string(prefx, rand(item_names[navimap.nodes[curr_s[end][1:2]]])), description))
                end

                push!(cands, (string("there is $det ", rand(item_names[navimap.nodes[curr_s[end][1:2]]]), " in this intersection"), description))
            end


            next = getlocation(navimap, curr_s[end], 1)
            if haskey(navimap.edges[(curr_s[end][1], curr_s[end][2])], (next[1], next[2])) && (length(cons) == 0 || description in cons)
                wpatrn, fpatrn = navimap.edges[(curr_s[end][1], curr_s[end][2])][(next[1], next[2])]
                for verb in ["you are now facing the ", "you are now seeing the ", "you should see "]
                    for flr in vcat(floor_names[fpatrn], ColorMapping[fpatrn])
                        cors = [" path", " hall", " hallway", " alley", " corridor", "", " floor", " flooring"]
                        if flr == "flower" || flr == "octagon" || flr == "pink-flowered" || flr == "flowered" || flr == "rose"
                            push!(cors, " carpet")
                        end
                        for cor in cors
                            push!(cands, (string(verb, flr, cor), visual_t))
                        end
                    end
                end
            end
        end
        
        push!(insl, ([curr_s[end]], rand(cands)))
        return insl
    end
    return insl
end
