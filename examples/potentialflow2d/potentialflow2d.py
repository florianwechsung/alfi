from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class Potentialflow2DProblem(NavierStokesProblem):
    """ 
    This implements the numerical example 1 from https://arxiv.org/pdf/2007.04012.pdf 


    h = x^3 - 3*x*y*y
    u = ∇h
    p = −(|∇h|^2)/2 + 14/5
    f = 0
    wind = u

    with 

    """

    def __init__(self, baseN, diagonal=None):
        super().__init__()
        self.baseN = baseN
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1,
                             distribution_parameters=distribution_parameters,
                             diagonal=self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), "on_boundary")]
        return bcs

    def has_nullspace(self): return True

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        h = x*x*x - 3*x*y*y
        driver = grad(h)
        return driver

    def char_length(self): return 1.0

    def relaxation_direction(self): return "0+:1-"

    def wind(self, V):
        return self.driver(V.ufl_domain())


if __name__ == "__main__":


    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    args, _ = parser.parse_known_args()
    problem = Potentialflow2DProblem(args.baseN, args.diagonal)
    solver = get_solver(args, problem)

    start = 250
    end = 10000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 50, 100, 150, 200]# + list(range(start, end+step, step))
    results = run_solver(solver, res, args)
