from pprint import pprint
from ldc2d.ldc2d import TwoDimLidDrivenCavityProblem
from ldc3d.ldc3d import ThreeDimLidDrivenCavityProblem
from bfs2d.bfs2d import TwoDimBackwardsFacingStepProblem
from bfs3d.bfs3d import ThreeDimBackwardsFacingStepProblem
from planarlattice2d.planarlattice2d import PlanarLattice2DProblem
from potentialflow2d.potentialflow2d import Potentialflow2DProblem
from superposition2d.superposition2d import Superposition2DProblem
from alfi import get_default_parser, get_solver, run_solver
import os

parser = get_default_parser()
parser.add_argument("--problem", type=str, required=True,
                    choices=["ldc2d", "bfs2d", "ldc3d", "bfs3d"])
parser.add_argument("--diagonal", type=str, default="left",
                    choices=["left", "right", "crossed"])
parser.add_argument("--mesh", type=str)
parser.add_argument("--nref-start", type=int, required=True)
parser.add_argument("--nref-end", type=int, required=True)
parser.add_argument("--re-max", type=int, default=10000)
parser.add_argument("--singular", dest="singular", default=False,
                    action="store_true")
args, _ = parser.parse_known_args()
from_zero_each_time = False
if args.problem == "ldc2d":
    problem = TwoDimLidDrivenCavityProblem(args.baseN, args.diagonal, regularised=not args.singular)
elif args.problem == "bfs2d":
    problem = TwoDimBackwardsFacingStepProblem(args.mesh)
elif args.problem == "ldc3d":
    problem = ThreeDimLidDrivenCavityProblem(args.baseN)
elif args.problem == "bfs3d":
    problem = ThreeDimBackwardsFacingStepProblem(args.mesh)
elif args.problem == "planarlattice2d":
    problem = PlanarLattice2DProblem(args.mesh)
    from_zero_each_time = True
elif args.problem == "potentialflow2d":
    problem = Potentialflow2DProblem(args.mesh)
    from_zero_each_time = True
elif args.problem == "superposition2d":
    problem = Superposition2DProblem(args.mesh)
    from_zero_each_time = True
else:
    raise NotImplementedError

start = 200
end = 10000
step = 100
res = [1, 10, 100] + list(range(start, end+step, step))
res = [r for r in res if r <= args.re_max]
if args.problem in ["bfs2d", "bfs3d"]:
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
    results[nref] = {re: results_temp[re] for re in tableres if re in results_temp.keys()}
    comm = solver.mesh.comm

os.sys.stdout.flush()
if comm.rank == 0:
    table = [["nref\t", "dofs\t"] + tableres]
    for nref in nrefs:
        dofstr = ("%.2e" % dofs[nref]).replace("e+0", r"\times 10^")
        line = ["%i" % nref, "$%s$" % dofstr]
        for re in tableres:
            try:
                avg_ksp_iter = float(results[nref][re]["linear_iter"]
                                     / results[nref][re]["nonlinear_iter"])
            except:
                avg_ksp_iter = float('nan')
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
            try:
                avg_ksp_iter = float(results[nref][re]["time"]*60)
            except:
                avg_ksp_iter = float('nan')
            line.append(avg_ksp_iter)
        table.append(line)
    print(" \\\\\n".join(["\t& ".join(map(rnd, line)) for line in table]) + "\\\\")
