from firedrake import *
from firedrake.petsc import PETSc
#PETSc.Sys.popErrorHandler()
from mpi4py import MPI
from alfi import *

import os
import shutil

class TwoDimLidDrivenCavityProblem(NavierStokesProblem):
    def __init__(self, baseN,diagonal=None,regularised=True):
        super().__init__()
        self.baseN = baseN
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2,
                             distribution_parameters=distribution_parameters)#,
#                             diagonal=self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(0), Constant((0., 0.)), [1, 2, 3])]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        if self.regularised:
            driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), 0])
        else:
            driver = as_vector([(0.25*y*y), 0])
        return driver


    def interpolate_initial_guess(self, z):
        w_expr = self.driver(z.ufl_domain())
        z.sub(0).interpolate(w_expr)

    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    args, _ = parser.parse_known_args()

    if args.clear:
        shutil.rmtree("checkpoint", ignore_errors=True)
    if args.time:
        PETSc.Log.begin()
        
    problem = TwoDimLidDrivenCavityProblem(args.baseN)
    solver = get_solver(args, problem)
    problem.interpolate_initial_guess(solver.z)
    File("u.pvd").write(solver.z.split()[0])
    comm = solver.mesh.mpi_comm()

    pvdf = File("output/velocity.pvd")
    os.makedirs("checkpoint", exist_ok=True)
    results = []
#    res = [100,200]
    res = [100]
    
#    for i in range(10):
        dt = 0.0001
        warning("dt=%.4f" % dt)
        results=run_solver(solver,res,dt,args)
#        print("rob checked",np.shape(run_solver(solver,res,dt,args)))
#        (z, info_dict) =run_solver(solver,res,dt,args)
#        pvdf.write(solver.visprolong(solver.z.split()[0]), time=i)

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
