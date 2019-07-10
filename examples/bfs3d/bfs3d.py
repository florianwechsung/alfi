from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np
import os


class ThreeDimBackwardsFacingStepProblem(NavierStokesProblem):
    def __init__(self, msh):
        self.msh = msh
        super().__init__()

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/" + self.msh,
                    distribution_parameters=distribution_parameters)
        return base

    @staticmethod
    def poiseuille_flow(domain):
        (x, y, z) = SpatialCoordinate(domain)
        return as_vector([16*(2-y)*(y-1)*z*(1-z)*(y>1), 0, 0])

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.poiseuille_flow(Z.mesh()), 1),
               DirichletBC(Z.sub(0), Constant((0., 0., 0)), 3)]
        return bcs

    def has_nullspace(self):
        return False

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":


    parser = get_default_parser()
    parser.add_argument("--mesh", type=str, default="coarse09.msh",
                        choices=["coarse%i.msh" % i for i in [13, 35, 45]])
    args, _ = parser.parse_known_args()
    problem = ThreeDimBackwardsFacingStepProblem(args.mesh)
    solver = get_solver(args, problem)

    start = 250
    end = 1000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 100, 200] + list(range(start, end+step, step))
    results = run_solver(solver, res, args)
