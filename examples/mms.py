from pprint import pprint
from mmsldc2d.mmsldc2d import TwoDimLidDrivenCavityMMSProblem
from alfi import get_default_parser, get_solver, run_solver
import os
from firedrake import *
import numpy as np

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
args, _ = parser.parse_known_args()

if args.dim == 2:
    problem = TwoDimLidDrivenCavityMMSProblem(args.baseN)
else:
    raise NotImplementedError

res = [1, 10, 100]#, 200, 500]
results = {}
for re in res:
    results[re] = {}
    for s in ["velocity", "pressure"]:
        results[re][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    solver = get_solver(args, problem)
    comm = solver.mesh.comm
    hs.append(1/((solver.mesh.coordinates.function_space().dim()/2)**(1./args.dim)-1))

    for re in res:
        problem.Re.assign(re)

        (z, info_dict) = solver.solve(re)
        z = solver.z
        u, p = z.split()
        Z = z.function_space()

        # uviz = solver.visprolong(u)
        # (u_, p_) = problem.actual_solution(uviz.function_space())
        # File("output/u-re-%i-nref-%i.pvd" % (re, nref)).write(uviz.interpolate(uviz))
        # File("output/uerr-re-%i-nref-%i.pvd" % (re, nref)).write(uviz.interpolate(uviz-u_))
        # File("output/uex-re-%i-nref-%i.pvd" % (re, nref)).write(uviz.interpolate(u_))
        (u_, p_) = problem.actual_solution(Z)
        # File("output/perr-re-%i-nref-%i.pvd" % (re, nref)).write(Function(Z.sub(1)).interpolate(p-p_))
        veldiv = norm(div(u))
        pressureintegral = assemble(p_ * dx)
        uerr = errornorm(u_, u, norm_type="H1")
        perr = errornorm(p_, p)
        pinterp = p.copy(deepcopy=True).interpolate(p_)
        pinterperror = errornorm(p_, pinterp)
        pintegral = assemble(p*dx)

        results[re]["velocity"].append(uerr)
        results[re]["pressure"].append(perr)
        if comm.rank == 0:
            print("|div(u_h)| = ", veldiv)
            print("p_exact * dx = ", pressureintegral)
            print("p_approx * dx = ", pintegral)

if comm.rank == 0:
    for re in res:
        print("Results for Re =", re)
        print("|u-u_h|", results[re]["velocity"])
        print("convergence orders:", convergence_orders(results[re]["velocity"]))
        print("|p-p_h|", results[re]["pressure"])
        print("convergence orders:", convergence_orders(results[re]["pressure"]))
    print("gamma =", args.gamma)
    print("h =", hs)

    for re in res:
        print("%%Re = %i" % re)
        print("\\pgfplotstableread[col sep=comma, row sep=\\\\]{%%")
        print("h,error_v,error_p\\\\")
        for i in range(len(hs)):
            print(",".join(map(str, [hs[i], results[re]["velocity"][i], results[re]["pressure"][i]])) + "\\\\")
        print("}\\\\re%i" % re)
