from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class DfgBenchmarkProblem(NavierStokesProblem):


    def __init__(self):
        super().__init__()

    # def mesh_hierarchy(self, hierarchy, nref, callbacks, distribution_parameters):
    #     if hierarchy != "uniform":
    #         raise NotImplemtedError("Pipe problem currently just works for uniform refinements")

    #     warning(RED % "Using cached mesh for reproducability!")
    #     mh = OpenCascadeMeshHierarchy(
    #         "meshes/pipe%id.step" % self.dim, element_size=self.element_size,
    #         levels=nref, order=self.order, cache=False, verbose=True,
    #         distribution_parameters=distribution_parameters,
    #         callbacks=callbacks, project_refinements_to_cad=False,
    #         reorder=True, cache=True,
    #         gmsh="gmsh -algo del%id -optimize_netgen 10 -smooth 10 -format msh2" % self.dim
    #     )
    #     return mh

    def mesh(self, distribution_parameters):
        return Mesh("dfg.msh", distribution_parameters=distribution_parameters)

    def bcs(self, Z):
        x, y = SpatialCoordinate(Z.ufl_domain())
        U_inflow = 0.3
        inflow_profile = as_vector([4.0 * U_inflow * y * (0.41 - y) / 0.41 ** 2, 0.0])
        bcs = [DirichletBC(Z.sub(0), inflow_profile, (1,)),
               DirichletBC(Z.sub(0), Constant((0, 0)), (2, 3))]
        return bcs

    def has_nullspace(self): return False

    def char_length(self): return 0.1

    def char_velocity(self): return 0.2

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":


    parser = get_default_parser()
    args, _ = parser.parse_known_args()
    problem = DfgBenchmarkProblem()
    solver = get_solver(args, problem)

    res = [1, 10, 20, 50]#, 100, 200, 400, 500]
    results = run_solver(solver, res, args)
