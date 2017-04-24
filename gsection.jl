function goldensection(f,n; dxmin=0.1, accel=golden, history=[], verbose=false)

    function feval(x)           # so we don't repeat function evaluations
        for (k,v) in history
            if isapprox(x,k)
                return v
            end
        end
        fx = f(x)
        push!(history, (x,fx))
        return fx
    end

    function setindex(x,v,d)    # non-mutating setindex
        y = copy(x)
        y[d] = v
        return y
    end

    x0 = zeros(n)               # initial point
    f0 = feval(x0)              # initial value
    dx = ones(n)                # step sizes
    df = Inf * ones(n)          # expected gains
    while maximum(abs(dx)) >= dxmin
        i = indmax(df)
        x1 = setindex(x0,x0[i]+dx[i],i)
        f1 = feval(x1)
        if verbose; debug((:f0,f0,:x0,x0,:f1,f1,:x1,x1,:dx,dx,:df,df)); end
        isnan(f1) && (f1=f0+df[i])
        if f1 < f0
            dx[i] = accel * dx[i]
            df[i] = accel * (f0-f1)
            x0,f0 = x1,f1
            for j = 1:length(df)
                if abs(dx[j]) < dxmin * accel
                    dx[j] = sign(dx[j]) * dxmin * accel
                    df[j] = max(df[j],0) # max(df[j],-1-df[j])
                end
            end
        else
            dx[i] = -dx[i] / accel
            df[i] = (f1-f0) / accel
            if abs(dx[i]) < dxmin
                df[i] = -1 # -1-df[i]
            end
        end
    end
    return (f0,x0)
end
