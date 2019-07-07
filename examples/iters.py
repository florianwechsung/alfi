from pprint import pprint
from ldc2d.ldc2d import TwoDimLidDrivenCavityProblem
from ldc3d.ldc3d import ThreeDimLidDrivenCavityProblem
from bfs2d.bfs2d import TwoDimBackwardsFacingStepProblem
from alfi import get_default_parser, get_solver, run_solver
import os

parser = get_default_parser()
parser.add_argument("--problem", type=str, required=True,
                    choices=["ldc2d", "bfs2d"])
parser.add_argument("--diagonal", type=str, default="left",
                    choices=["left", "right", "crossed"])
parser.add_argument("--mesh", type=str)
parser.add_argument("--nref-start", type=int)
parser.add_argument("--nref-end", type=int)
args, _ = parser.parse_known_args()

if args.problem == "ldc2d":
    problem = TwoDimLidDrivenCavityProblem(args.baseN, args.diagonal)
elif args.problem == "bfs2d":
    problem = TwoDimBackwardsFacingStepProblem(args.mesh)
elif args.problem == "ldc3d":
    problem = ThreeDimLidDrivenCavityProblem(args.baseN)
else:
    raise NotImplementedError

start = 200
end = 10000
step = 100
res = [1, 10, 100] + list(range(start, end+step, step))
if args.problem == "bfs2d":
    res = list(sorted(res + [50, 150, 250, 350]))
results = {}
nrefs = range(args.nref_start, args.nref_end+1)
tableres = [i for i in [10, 100, 1000, 5000, 10000] if i <= max(res)]
dofs = {}
for nref in nrefs:
    args.nref = nref
    solver = get_solver(args, problem)
    dofs[nref] = solver.Z.dim()
    results_temp = run_solver(solver, res, args)
    results[nref] = {re: results_temp[re] for re in tableres}
    comm = solver.mesh.comm

os.sys.stdout.flush()
if comm.rank == 0:
    table = [["nref\t", "dofs\t"] + tableres]
    for nref in nrefs:
        dofstr = ("%.2e" % dofs[nref]).replace("e+0", r"\times 10^")
        line = ["%i" % nref, "$%s$" % dofstr]
        for re in tableres:
            avg_ksp_iter = float(results[nref][re]["linear_iter"]
                                 / results[nref][re]["nonlinear_iter"])
            line.append(avg_ksp_iter)
        table.append(line)

    def rnd(i):
        if isinstance(i, str) or isinstance(i, int):
            return str(i)
        return "%.2f" % i
    print(" \\\\\n".join(["\t& ".join(map(rnd, line)) for line in table]) + "\\\\")

    table = [["nref", "dofs\t"] + tableres]
    for nref in nrefs:
        dofstr = ("%.2e" % dofs[nref]).replace("e+0", r"\times 10^")
        line = ["%i" % nref, "$%s$" % dofstr]
        for re in tableres:
            avg_ksp_iter = float(results[nref][re]["time"]*60)
            line.append(avg_ksp_iter)
        table.append(line)
    print(" \\\\\n".join(["\t& ".join(map(rnd, line)) for line in table]) + "\\\\")
