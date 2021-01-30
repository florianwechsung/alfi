from firedrake import *
from firedrake.petsc import PETSc
from alfi import *
import numpy as np
import os



class TwoDimLidDrivenCavityMMSProblem(NavierStokesProblem):
    def __init__(self, baseN, Re=Constant(1), Repres=None):
        super().__init__()
        self.baseN = baseN
        self.Re = Re
        if Repres is None:
            self.Repres = Re
        else:
            self.Repres = Repres

    def mesh(self, distribution_parameters):
        # base = RectangleMesh(self.baseN, self.baseN, 2, 2,
        #                      distribution_parameters=distribution_parameters)

        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
                    distribution_parameters=distribution_parameters)
        
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.actual_solution(Z)[0], [4]),
               DirichletBC(Z.sub(0), Constant((0., 0.)), [1, 2, 3])]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.actual_solution(z)[0]
        z.sub(0).interpolate(w_expr)

    def char_length(self): return 2.0

    def actual_solution(self, Z):
        # Taken from 'EFFECTS OF GRID STAGGERING ON NUMERICAL SCHEMES - Shih, Tan, Hwang'
        X = SpatialCoordinate(Z.mesh())
        (x, y) = X
        """ Either implement the form in the paper by Shih, Tan, Hwang and then
        use ufl to rescale, or just implement the rescaled version directly """
        f = x**4 - 2 * x**3 + x**2
        g = y**4 - y**2

        from ufl.algorithms.apply_derivatives import apply_derivatives
        df = apply_derivatives(grad(f)[0])
        dg = apply_derivatives(grad(g)[1])
        ddg = apply_derivatives(grad(dg)[1])
        dddg = apply_derivatives(grad(ddg)[1])

        F = 0.2 * x**5 - 0.5 * x**4 + (1./3.) * x**3
        F1 = -4*x**6 + 12*x**5 - 14*x**4 + 8 * x**3 - 2*x**2
        F2 = 0.5 * f**2
        G1 = -24 * y**5+8*y**3-4*y
        u = 8 * f * dg
        v = -8 * df * g
        p = (8./self.Re) * (F * dddg + df*dg) + 64 * F2 * (g*ddg-dg**2)
        u = replace(u, {X: 0.5 * X})
        v = replace(v, {X: 0.5 * X})
        p = replace(p, {X: 0.5 * X})
        p = p - 0.25 * assemble(p*dx)
        # u = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2)
        # v = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
        # p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5*self.Re)
        # p = p - (-(1408./33075.) + 8./(5*self.Re))
        driver = as_vector([u, v])
        return (driver, p)

    def rhs(self, Z):
        (u, p) = self.actual_solution(Z)
        nu = self.char_length() * self.char_velocity() / self.Re
        f1 = -nu * div(2*sym(grad(u))) + dot(grad(u), u) + grad(p)
        f2 = -div(u)
        return f1, f2
