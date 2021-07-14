# Run this script from the command line:
# $ julia sekira ARGS...

# To see available and required ARGS, run
# $ julia sekira --help
# or see
# julia> ?Sekira.plateau
using Sekira
#H = HubbardMom1D(BoseFS((0, 0, 0, 7, 0, 0, 0)); u=6)
#H = HubbardMom1D(BoseFS((0, 0, 0, 0, 10, 0, 0, 0, 0, 0)); u=6)
add = BoseFS(ntuple(i -> ifelse(i==10, 10, 0), 20))
H = HubbardMom1D(add; u=0.5)

Sekira.plateau(H, ARGS)
