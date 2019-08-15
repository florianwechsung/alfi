from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np
import os


class TwoDimBackwardsFacingStepProblem(NavierStokesProblem):
    def __init__(self, msh):
        self.msh = msh
        super().__init__()

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/" + self.msh,
                    distribution_parameters=distribution_parameters)
        return base

    @staticmethod
    def poiseuille_flow(domain):
        (x, y) = SpatialCoordinate(domain)
        return as_vector([4 * (2-y)*(y-1)*(y>1), 0])

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.poiseuille_flow(Z.mesh()), 1),
               DirichletBC(Z.sub(0), Constant((0., 0.)), 2)]
        return bcs

    def has_nullspace(self):
        return False

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":


    parser = get_default_parser()
    parser.add_argument("--mesh", type=str, default="coarse09.msh",
                        choices=["coarse03.msh", "coarse04.msh",
                                 "coarse06.msh", "coarse09.msh",
                                 "coarse12.msh"])
    args, _ = parser.parse_known_args()
    problem = TwoDimBackwardsFacingStepProblem(args.mesh)
    solver = get_solver(args, problem)

    start = 250
    end = 10000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 100] # + list(range(start, end+step, step))
    results = run_solver(solver, res, args)
