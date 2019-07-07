from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class ThreeDimLidDrivenCavityProblem(NavierStokesProblem):
    def __init__(self, baseN):
        super().__init__()
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, self.baseN, 2, 2, 2,
                            distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(0), Constant((0., 0., 0.)), [1, 2, 3, 5, 6])]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y, z) = SpatialCoordinate(domain)
        driver = as_vector([x*x*(2-x)*(2-x)*z*z*(2-z)*(2-z)*(0.25*y*y), 0, 0])
        return driver

    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":


    parser = get_default_parser()
    args, _ = parser.parse_known_args()
    problem = ThreeDimLidDrivenCavityProblem(args.baseN)
    solver = get_solver(args, problem)

    start = 250
    end = 10000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 100]  # + list(range(start, end+step, step))
    results = run_solver(solver, res, args)
