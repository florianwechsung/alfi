from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np


class PlanarLattice2DProblem(NavierStokesProblem):
    """ 
    This implements the numerical example 2 from
    https://arxiv.org/pdf/2007.04012.pdf 
    i.e.

        u = (sin(2πx)*sin(2πy), cos(2πx)*cos(2πy))
        p = (cos(4πx)−cos(4πy))/4
        f = − μ∆u
        wind = u

    """

    def __init__(self, baseN, diagonal=None):
        super().__init__()
        self.baseN = baseN
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.nu = None

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
        driver = as_vector([sin(2*pi*x)*sin(2*pi*y), cos(2*pi*x)*cos(2*pi*y)])
        return driver

    def char_length(self): return 1.0

    def relaxation_direction(self): return "0+:1-"

    def rhs(self, Z):
        u = self.driver(Z.ufl_domain())
        f1 = -self.nu * div(2*sym(grad(u)))# + dot(grad(u), u) + grad(p)
        f2 = Constant(0)#-div(u)
        return f1, f2

    def wind(self, V):
        return self.driver(V.ufl_domain())


if __name__ == "__main__":


    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    args, _ = parser.parse_known_args()
    problem = PlanarLattice2DProblem(args.baseN, args.diagonal)
    solver = get_solver(args, problem)

    start = 250
    end = 10000
    step = 250
    res = [0, 1, 10, 100] + list(range(start, end+step, step))
    res = [1, 10, 50, 100, 150, 200]# + list(range(start, end+step, step))
    results = run_solver(solver, res, args, from_zero_each_time=True)
