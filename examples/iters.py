from pprint import pprint
from ldc2d.ldc2d import TwoDimLidDrivenCavityProblem
from alfi import get_default_parser, get_solver, run_solver
import os

parser = get_default_parser()
parser.add_argument("--problem", type=str, required=True,
                    choices=["ldc2d"])
args, _ = parser.parse_known_args()

if args.problem == "ldc2d":
    problem = TwoDimLidDrivenCavityProblem(args.baseN)
else:
    raise NotImplementedError

start = 200
end = 10000
step = 100
res = [1, 10, 100] + list(range(start, end+step, step))

results = {}
nrefs = range(1, 4)
tableres = [i for i in [1, 10, 100, 1000, 5000, 10000] if i <= max(res)]
for nref in nrefs:
    args.nref = nref
    solver = get_solver(args, problem)
    results_temp = run_solver(solver, res, args)
    results[nref] = {re: results_temp[re] for re in tableres}
    comm = solver.mesh.comm

os.sys.stdout.flush()
if comm.rank == 0:
    table = [["nref\Re\t"] + tableres]
    for nref in nrefs:
        line = ["%i\t" % nref]
        for re in tableres:
            avg_ksp_iter = float(results[nref][re]["linear_iter"]
                                 / results[nref][re]["nonlinear_iter"])
            line.append(avg_ksp_iter)
        table.append(line)

    def rnd(i):
        if isinstance(i, str) or isinstance(i, int):
            return str(i)
        return "%.2f" % i
    print(" \\\\\n".join([" \t& ".join(map(rnd, line)) for line in table]))
