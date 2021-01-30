from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class TwoDimLidDrivenCavityProblem(NavierStokesProblem):
    def __init__(self, baseN, diagonal=None, regularised=True):
        super().__init__()
        self.baseN = baseN
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2,
                             distribution_parameters=distribution_parameters,
                             diagonal=self.diagonal)
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

    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":


    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    args, _ = parser.parse_known_args()
    problem = TwoDimLidDrivenCavityProblem(args.baseN, args.diagonal)
    solver = get_solver(args, problem)

    start = 250
    end = 10000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 50, 100, 150, 200]# + list(range(start, end+step, step))
    results = run_solver(solver, res, args)
