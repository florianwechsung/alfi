from firedrake import *
from firedrake.petsc import PETSc
#PETSc.Sys.popErrorHandler()
from mpi4py import MPI
from alfi import *

import os
import shutil

class ChannelFlow(NavierStokesProblem):
    def __init__(self):
        super().__init__()

    def mesh(self, distribution_parameters):
        base = Mesh("coarse12.msh", distribution_parameters=distribution_parameters)
        return base

    @staticmethod
    def poiseuille_flow(domain):
        (x, y) = SpatialCoordinate(domain)
        return as_vector([4 *y* (1-y), 0])

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.poiseuille_flow(Z.mesh()), 1),
                DirichletBC(Z.sub(0), Constant((0., 0.)), 3)]
        return bcs

    def has_nullspace(self): return False

    def IC(self,domain):
        (x,y)=SpatialCoordinate(domain)
        IC = as_vector([0,0])
        return IC

    def interpolate_initial_guess(self, z):
        w_expr =self.IC(z.ufl_domain())
        z.sub(0).interpolate(w_expr)

    def relaxation_direction(self): return "0+:1-"

if __name__ == "__main__":

    parser = get_default_parser()
    args, _ = parser.parse_known_args()

    if args.clear:
        shutil.rmtree("checkpoint", ignore_errors=True)
    if args.time:
        PETSc.Log.begin()
        
    problem = ChannelFlow()
    solver = get_solver(args, problem)
    problem.interpolate_initial_guess(solver.z)
    File("u.pvd").write(solver.z.split()[0])
    comm = solver.mesh.mpi_comm()

    pvdf = File("output/velocity.pvd")
    os.makedirs("checkpoint", exist_ok=True)
    results = []
    re=100

#    dt = 0.0001
    dt = 0.01
    warning("dt=%.4f" % dt)
    results=run_solver(solver,re,50,dt,args)

    if comm.rank == 0:
        for info_dict in results:
            print(info_dict)

    if args.time:
        if comm.rank == 0:
            print(BLUE % "Some performance info:")
        events = ["MatMult", "MatSolve", "PCSetUp", "PCApply", "PCPATCHSolve", "PCPATCHApply", "KSPSolve_FS_0",  "KSPSolve_FS_Low", "KSPSolve", "SNESSolve", "ParLoopExecute", "ParLoopCells", "SchoeberlProlong", "inject", "prolong", "restrict", "MatFreeMatMult", "MatFreeMatMultTranspose"]
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
            print(BLUE % ("%i \t % 5.1fs \t % 4.2fs \t %i" % (args.nref, time, 1000*time/solver.Z.dim(), solver.Z.dim())))
