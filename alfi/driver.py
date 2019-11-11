from alfi.solver import ConstantPressureSolver, ScottVogeliusSolver
from mpi4py import MPI
from firedrake.petsc import PETSc
from firedrake import *
import os
import shutil


def get_default_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nref", type=int, default=1)
    parser.add_argument("--nref-vis", type=int, default=0)
    parser.add_argument("--baseN", type=int, default=16)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--stabilisation-weight", type=float, default=None)
    parser.add_argument("--solver-type", type=str, default="almg",
                        choices=["lu", "allu", "almg", "alamg", "simple", "lsc"])
    parser.add_argument("--patch", type=str, default="star",
                        choices=["star", "macro"])
    parser.add_argument("--patch-composition", type=str, default="additive",
                        choices=["additive", "multiplicative"])
    parser.add_argument("--mh", type=str, default="uniform",
                        choices=["uniform", "bary", "uniformbary"])
    parser.add_argument("--stabilisation-type", type=str, default=None,
                        choices=["none", "burman", "gls", "supg"])
    parser.add_argument("--discretisation", type=str, required=True,
                        choices=["pkp0", "sv"])
    parser.add_argument("--gamma", type=float, default=1e4)
    parser.add_argument("--clear", dest="clear", default=False,
                        action="store_true")
    parser.add_argument("--time", dest="time", default=False,
                        action="store_true")
    parser.add_argument("--mkl", dest="mkl", default=False,
                        action="store_true")
    parser.add_argument("--checkpoint", dest="checkpoint", default=False,
                        action="store_true")
    parser.add_argument("--paraview", dest="paraview", default=False,
                        action="store_true")
    parser.add_argument("--restriction", dest="restriction", default=False,
                        action="store_true")
    parser.add_argument("--rebalance", dest="rebalance", default=False,
                        action="store_true")
    parser.add_argument("--high-accuracy", dest="high_accuracy", default=False,
                        action="store_true")
    parser.add_argument("--smoothing", type=int, default=None)
    return parser


def get_solver(args, problem, hierarchy_callback=None):
    solver_t = {"pkp0": ConstantPressureSolver,
                "sv": ScottVogeliusSolver}[args.discretisation]
    solver = solver_t(
        problem,
        solver_type=args.solver_type,
        stabilisation_type=args.stabilisation_type,
        nref=args.nref,
        k=args.k,
        gamma=args.gamma,
        nref_vis=args.nref_vis,
        patch=args.patch,
        use_mkl=args.mkl,
        supg_method="shakib",
        stabilisation_weight=args.stabilisation_weight,
        hierarchy=args.mh,
        patch_composition=args.patch_composition,
        restriction=args.restriction,
        smoothing=args.smoothing,
        rebalance_vertices=args.rebalance,
        high_accuracy=args.high_accuracy,
        hierarchy_callback=hierarchy_callback,
    )
    return solver


def performance_info(comm, solver):
        if comm.rank == 0:
            print(BLUE % "Some performance info:")
        events = ["MatMult", "MatSolve", "PCSetUp", "PCApply", "PCPATCHSolve", "PCPATCHApply", "KSPSolve_FS_0",  "KSPSolve_FS_Low", "KSPSolve", "SNESSolve", "ParLoopExecute", "ParLoopCells", "SchoeberlProlong", "SchoeberlRestrict", "inject", "prolong", "restriction", "MatFreeMatMult", "MatFreeMatMultTranspose", "DMPlexRebalanceSharedPoints", "PCPatchComputeOp", "PCPATCHScatter"]
        perf = dict((e, PETSc.Log.Event(e).getPerfInfo()) for e in events)
        perf_reduced = {}
        for k, v in perf.items():
            perf_reduced[k] = {}
            for kk, vv in v.items():
                perf_reduced[k][kk] = comm.allreduce(vv, op=MPI.SUM) / comm.size
        perf_reduced_sorted = [(k, v) for (k, v) in sorted(perf_reduced.items(), key=lambda d: -d[1]["time"])]
        if comm.rank == 0:
            for k, v in perf_reduced_sorted:
                print(GREEN % (("%s:" % k).ljust(30) + "Time = % 6.2fs, Time/1kdofs = %.2fs" % (v["time"], 1000*v["time"]/solver.Z.dim())))
            time = perf_reduced_sorted[0][1]["time"]
            print(BLUE % ("% 5.1fs \t % 4.2fs \t %i" % (time, 1000*time/solver.Z.dim(), solver.Z.dim())))


def run_solver(solver, res, args):
    if args.time:
        PETSc.Log.begin()
    problemsize = solver.Z.dim()
    outdir = "output/%i/" % problemsize
    chkptdir = "checkpoint/%i/" % problemsize
    if args.clear:
        shutil.rmtree(chkptdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
    comm = solver.mesh.mpi_comm()
    comm.Barrier()
    if args.paraview:
        pvdf = File(outdir + "velocity.pvd")
    if args.checkpoint:
        os.makedirs(chkptdir, exist_ok=True)
    results = {}
    for re in res:
        try:
            with DumbCheckpoint(chkptdir + "nssolution-Re-%s" % (re), mode=FILE_READ) as checkpoint:
                checkpoint.load(solver.z, name="up_%i" % re)
        except:
            (z, info_dict) = solver.solve(re)
            results[re] = info_dict
            if args.checkpoint:
                with DumbCheckpoint(chkptdir + "nssolution-Re-%s" % (re), mode=FILE_UPDATE) as checkpoint:
                    checkpoint.store(solver.z, name="up_%i" % re)
        if args.paraview:
            pvdf.write(solver.visprolong(solver.z.split()[0]), time=re)

    if comm.rank == 0:
        for re in results:
            print(results[re])
    if args.time:
        performance_info(comm, solver)
    return results
