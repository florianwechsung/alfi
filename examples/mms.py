from pprint import pprint
from mmsldc2d.mmsldc2d import TwoDimLidDrivenCavityMMSProblem
from mmsldc3d.mmsldc3d import ThreeDimLidDrivenCavityMMSProblem
from alfi import get_default_parser, get_solver, run_solver
import os
from firedrake import *
import numpy as np
import inflect
eng = inflect.engine()

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
args, _ = parser.parse_known_args()

if args.dim == 2:
    problem = TwoDimLidDrivenCavityMMSProblem(args.baseN)
elif args.dim == 3:
    problem = ThreeDimLidDrivenCavityMMSProblem(args.baseN)
else:
    raise NotImplementedError

res = [1, 9, 10, 50, 90, 100, 400, 500, 900, 1000]
results = {}
for re in res:
    results[re] = {}
    for s in ["velocity", "velocitygrad", "pressure", "divergence", "relvelocity", "relvelocitygrad", "relpressure"]:
        results[re][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    solver = get_solver(args, problem)
    mesh = solver.mesh
    h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh))
    with h.dat.vec_ro as w:
        hs.append((w.max()[1], w.sum()/w.getSize()))
    # h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh)).vector().max()
    # hs.append(h)
    comm = mesh.comm

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
        uerr = norm(u_-u)
        ugraderr = norm(grad(u_-u))
        perr = norm(p_-p)
        pinterp = p.copy(deepcopy=True).interpolate(p_)
        pinterperror = errornorm(p_, pinterp)
        pintegral = assemble(p*dx)

        results[re]["velocity"].append(uerr)
        results[re]["velocitygrad"].append(ugraderr)
        results[re]["pressure"].append(perr)
        results[re]["relvelocity"].append(uerr/norm(u_))
        results[re]["relvelocitygrad"].append(ugraderr/norm(grad(u_)))
        results[re]["relpressure"].append(perr/norm(p_))
        results[re]["divergence"].append(veldiv)
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

    for re in [10, 100, 500, 1000]:
        print("%%Re = %i" % re)
        print("\\pgfplotstableread[col sep=comma, row sep=\\\\]{%%")
        print("hmin,havg,error_v,error_vgrad, error_p,relerror_v, relerror_vgrad,relerror_p,div\\\\")
        for i in range(len(hs)):
            print(",".join(map(str, [hs[i][0], hs[i][1], results[re]["velocity"][i], results[re]["velocitygrad"][i], results[re]["pressure"][i], results[re]["relvelocity"][i], results[re]["relvelocitygrad"][i], results[re]["relpressure"][i], results[re]["divergence"][i]])) + "\\\\")
        def numtoword(num):
            return eng.number_to_words(num).replace(" ", "").replace("-","")
        name = "re" + numtoword(int(re)) \
            + "gamma" + numtoword(int(args.gamma)) \
            + args.discretisation.replace("0", "zero")
        print("}\\%s" % name)
