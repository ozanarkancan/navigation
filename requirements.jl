l = ["ArgParse", "JLD", "JSON", "DataStructures", "Logging"]

for p in l; Pkg.add(p); end

Pkg.clone("https://github.com/ozanarkancan/Knet.jl")
Pkg.checkout("Knet", "transpose")
Pkg.build("Knet")
