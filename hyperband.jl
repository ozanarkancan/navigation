function hyperband(getconfig, getloss, maxresource, reduction=3)
    @show smax = floor(Int, log(maxresource)/log(reduction))
    @show B = (smax + 1) * maxresource
    best = (Inf,)
    for s in smax:-1:0
        n = ceil(Int, (B/maxresource)*((reduction^s)/(s+1)))
        r = maxresource / (reduction^s)
        curr = halving(getconfig, getloss, n, r, reduction, s)
        if curr[1] < best[1]; (best=curr); end
    end
    return best
end

function halving(getconfig, getloss, n, r=1, reduction=3, s=round(Int, log(n)/log(reduction)))
    best = (Inf,)
    T = [ getconfig() for i=1:n ]
    for i in 0:s
        ni = floor(Int,n/(reduction^i))
        ri = r*(reduction^i)
        println((:s,s,:n,n,:r,r,:i,i,:ni,ni,:ri,ri,:T,length(T)))
        L = [ getloss(t, ri) for t in T ]
        l = sortperm(L); l1=l[1]
        L[l1] < best[1] && (best = (L[l1],ri,T[l1]); println("best1: $best"))
        T = T[l[1:floor(Int,ni/reduction)]]
    end
    println("best2: $best")
    return best
end
