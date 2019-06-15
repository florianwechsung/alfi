from firedrake import *
from firedrake.petsc import *
import numpy as np

from alfi.stabilisation import *
# from alfi.element import velocity_element, pressure_element
# from alfi.utilities import coarsen
from alfi.transfer import *
from alfi.bary import BaryMeshHierarchy, bary

import pprint
import sys
from datetime import datetime

class DGMassInv(PCBase):

    def initialize(self, pc):
        _, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        V = dmhooks.get_function_space(pc.getDM())
        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)
        massinv = assemble(Tensor(inner(u, v)*dx).inv)
        massinv.force_evaluation()
        self.massinv = massinv.petscmat
        self.nu = appctx["nu"]
        self.gamma = appctx["gamma"]

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.massinv.mult(x, y)
        scaling = float(self.nu) + float(self.gamma)
        y.scale(-scaling)

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError("Sorry!")


class NavierStokesSolver(object):

    def function_space(self, mesh, k):
        raise NotImplementedError

    def residual(self):
        raise NotImplementedError
    
    def update_wind(self, z):
        raise NotImplementedError

    def set_transfers(self):
        raise NotImplementedError

    def __init__(self, problem, nref=1, solver_type="almg",
                 stabilisation_type=None,
                 supg_method="shakib", supg_magic=9.0, gamma=10000, nref_vis=1,
                 k=5, patch="star", hierarchy="bary", use_mkl=False, stabilisation_weight=None):

        assert solver_type in {"almg", "allu", "lu"}, "Invalid solver type %s" % solver_type
        if stabilisation_type == "none":
            stabilisation_type = None
        assert stabilisation_type in {None, "gls", "supg", "burman"}, "Invalid stabilisation type %s" % stabilisation_type
        assert hierarchy in {"uniform", "bary", "uniformbary"}, "Invalid hierarchy type %s" % hierarchy
        assert patch in {"macro", "star"}, "Invalid patch type %s" % patch
        if hierarchy != "bary" and patch == "macro":
            raise ValueError("macro patch only makes sense with a BaryHierarchy")
        self.hierarchy = hierarchy

        self.problem = problem
        self.nref = nref
        self.solver_type = solver_type
        self.stabilisation_type = stabilisation_type
        self.patch = patch
        baseMesh = problem.mesh(problem.distribution_parameters)
        self.parallel = baseMesh.comm.size > 1
        self.tdim = baseMesh.topological_dimension()
        self.use_mkl = use_mkl


        def before(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+1)

        def after(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+2)

        if hierarchy == "bary":
            mh = BaryMeshHierarchy(baseMesh, nref, callbacks=(before, after),
                                   reorder=True, distribution_parameters=problem.distribution_parameters)
        elif hierarchy == "uniformbary":
            bmesh = Mesh(bary(baseMesh._plex), distribution_parameters={"partition": False})
            mh = MeshHierarchy(bmesh, nref, reorder=True, callbacks=(before, after),
                               distribution_parameters=problem.distribution_parameters)
        elif hierarchy == "uniform":
            mh = MeshHierarchy(baseMesh, nref, reorder=True, callbacks=(before, after),
                               distribution_parameters=problem.distribution_parameters)
        else:
            raise NotImplementedError("Only know bary, uniformbary and uniform for the hierarchy.")
        self.area = assemble(Constant(1, domain=mh[0])*dx)
        nu = Constant(1.0)
        self.nu = nu
        self.char_L = problem.char_length()
        self.char_U = problem.char_velocity()
        Re = self.char_L*self.char_U / nu
        self.Re = Re
        if not isinstance(gamma, Constant):
            gamma = Constant(gamma)
        self.gamma = gamma
        self.advect = Constant(0)

        mesh = mh[-1]
        uviss = []
 
        self.mesh = mesh
        Z = self.function_space(mesh, k)
        self.Z = Z
        comm = mesh.mpi_comm()
        if comm.size == 1:
            visbase = firedrake.Mesh(mesh._plex.clone(), dim=mesh.ufl_cell().geometric_dimension(),
                                     distribution_parameters=problem.distribution_parameters,
                                     reorder=True)
            vismh = MeshHierarchy(visbase, nref_vis)
            for vismesh in vismh:
                V = VectorFunctionSpace(vismesh, Z.sub(0).ufl_element())
                uviss.append(Function(V, name="VelocityRefined"))
            def visprolong(u):
                uviss[0].dat.data[:] = u.dat.data_ro
                if nref_vis == 0:
                    return uviss[0]
                uc = uviss[0]
                for i in range(nref_vis):
                    prolong(uc, uviss[i+1])
                    uc = uviss[i+1]
                return uviss[-1]
        else:
            def visprolong(u):
                return u

        self.visprolong = visprolong



        Zdim = self.Z.dim()
        size = comm.size
        if comm.rank == 0:
            print("Number of degrees of freedom: %s (avg %.2f per core)" % (Zdim, Zdim / size))
        z = Function(Z, name="Solution")
        z.split()[0].rename("Velocity")
        z.split()[1].rename("Pressure")
        self.z = z
        (u, p) = split(z)
        (v, q) = split(TestFunction(Z))

        bcs = problem.bcs(Z)
        nsp = problem.nullspace(Z)
        if nsp is not None and solver_type == "lu":
            """ Pin the pressure because LU sometimes fails for the saddle
            point problem with a nullspace """
            bcs.append(DirichletBC(Z.sub(1), Constant(0), None))
            if Z.mesh().comm.rank == 0:
                bcs[-1].nodes = np.asarray([0])
            else:
                bcs[-1].nodes = np.asarray([], dtype=np.int64)
            self.nsp = None
        else:
            self.nsp = nsp

        params = self.get_parameters()
        if mesh.mpi_comm().rank == 0:
            pprint.pprint(params)
            sys.stdout.flush()

        self.z_last = z.copy(deepcopy=True)


        F = self.residual()

        """ Stabilisation """
        wind = split(self.z_last)[0]
        rhs = problem.rhs(Z)
        if self.stabilisation_type in ["gls", "supg"]:
            if supg_method == "turek":
                self.stabilisation = TurekSUPG(Re, self.Z.sub(0), state=u, h=problem.mesh_size(u), magic=supg_magic, weight=stabilisation_weight)
            elif supg_method == "shakib":
                self.stabilisation = ShakibHughesZohanSUPG(1.0/nu, self.Z.sub(0), state=u, h=problem.mesh_size(u), magic=supg_magic, weight=stabilisation_weight)
            else:
                raise NotImplementedError

            Lu = -nu * div(2*sym(grad(u))) + dot(grad(u), u) + grad(p)
            Lv = -nu * div(2*sym(grad(v))) + dot(grad(v), wind) + grad(q)
            if rhs is not None:
                Lu -= rhs[0]
            k = Z.sub(0).ufl_element().degree()
            if self.stabilisation_type == "gls":
                self.stabilisation_form = self.stabilisation.form_gls(Lu, Lv, dx(degree=2*k))
            elif self.stabilisation_type == "supg":
                self.stabilisation_form = self.stabilisation.form(Lu, v, dx(degree=2*k))
            else:
                raise NotImplementedError
        elif self.stabilisation_type == "burman":
            self.stabilisation = BurmanStabilisation(self.Z.sub(0), state=wind, h=problem.mesh_size(u), weight=stabilisation_weight)
            self.stabilisation_form = self.stabilisation.form(u, v)
        else:
            self.stabilisation = None
            self.stabilisation_form = None

        if self.stabilisation_form is not None:
            F += (self.advect * self.stabilisation_form)

        if rhs is not None:
            F -= inner(rhs[0], v) * dx + inner(rhs[1], q) * dx

        appctx = {"nu": self.nu, "gamma": self.gamma}
        problem = NonlinearVariationalProblem(F, z, bcs=bcs)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters=params,
                                                 nullspace=nsp, options_prefix="ns_",
                                                 appctx=appctx)
        self.set_transfers()
        self.check_nograddiv_residual = True
        if self.check_nograddiv_residual:
            self.message(GREEN % "Checking residual without grad-div term")
            self.F_nograddiv = replace(F, {gamma: 0})
            self.F = F
            self.bcs = bcs

    def solve(self, re):
        self.z_last.assign(self.z)
        self.message(GREEN % ("Solving for Re = %s" % re))

        if re == 0:
            self.message(GREEN % ("Solving Stokes"))
            self.advect.assign(0)
            self.nu.assign(self.char_L*self.char_U)
        else:
            self.advect.assign(1)
            self.nu.assign(self.char_L*self.char_U/re)
        # self.gamma.assign(1+re)

        if self.stabilisation is not None:
            self.stabilisation.update(self.z.split()[0])
        start = datetime.now()
        self.solver.solve()
        end = datetime.now()

        if self.nsp is not None:
            # Hardcode that pressure integral is zero
            (u, p) = self.z.split()
            pintegral = assemble(p*dx)
            p.assign(p - Constant(pintegral/self.area))

        if self.check_nograddiv_residual:
            F_ngd = assemble(self.F_nograddiv)
            for bc in self.bcs:
                bc.zero(F_ngd)
            F = assemble(self.F)
            for bc in self.bcs:
                bc.zero(F)
            with F_ngd.dat.vec_ro as v_ngd, F.dat.vec_ro as v:
                self.message(BLUE % ("Residual without grad-div term: %.14e" % v_ngd.norm()))
                self.message(BLUE % ("Residual with grad-div term:    %.14e" % v.norm()))
        Re_linear_its = self.solver.snes.getLinearSolveIterations()
        Re_nonlinear_its = self.solver.snes.getIterationNumber()
        Re_time = (end-start).total_seconds() / 60
        self.message(GREEN % ("Time taken: %.2f min in %d iterations (%.2f Krylov iters per Newton step)" % (Re_time, Re_linear_its, Re_linear_its/float(Re_nonlinear_its))))
        info_dict = {
            "Re": re,
            "nu": self.nu.values()[0],
            "linear_iter": Re_linear_its,
            "nonlinear_iter": Re_nonlinear_its,
            "time": Re_time,
        }
        return (self.z, info_dict)

    def get_parameters(self):
        multiplicative = self.problem.relaxation_direction() is not None
        patchlu3d = "mkl_pardiso" if self.use_mkl else "umfpack"
        patchlu2d = "petsc"

        mg_levels_solver = {
            "ksp_type": "fgmres",
            "ksp_norm_type": "unpreconditioned",
            "ksp_max_it": 10 if self.tdim > 2 else 6,
            "ksp_convergence_test": "skip",
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch_pc_patch_save_operators": True,
            "patch_pc_patch_partition_of_unity": False,
            "patch_pc_patch_sub_mat_type": "seqaij" if self.tdim > 2 else "seqdense",
            "patch_pc_patch_sub_mat_type": "aij",
            "patch_pc_patch_local_type": "multiplicative" if multiplicative else "additive",
            "patch_pc_patch_statistics": False,
            "patch_pc_patch_symmetrise_sweep": multiplicative,
            "patch_pc_patch_precompute_element_tensors": True,
            "patch_sub_ksp_type": "preonly",
            "patch_sub_pc_type": "lu",
            "patch_sub_pc_factor_mat_solver_type": patchlu3d if self.tdim > 2 else patchlu2d
        }

        if self.patch == "star":
            if multiplicative:
                mg_levels_solver["patch_pc_patch_construct_type"] = "python"
                mg_levels_solver["patch_pc_patch_construct_python_type"] = "alfi.Star"
                mg_levels_solver["patch_pc_patch_construction_Star_sort_order"] = self.problem.relaxation_direction()
            else:
                mg_levels_solver["patch_pc_patch_construct_type"] = "star"
                mg_levels_solver["patch_pc_patch_construct_dim"] = 0
        elif self.patch == "macro":
            mg_levels_solver["patch_pc_patch_construct_type"] = "python"
            mg_levels_solver["patch_pc_patch_construct_python_type"] = "alfi.MacroStar"
            mg_levels_solver["patch_pc_patch_construction_MacroStar_sort_order"] = self.problem.relaxation_direction()
        else:
            raise NotImplementedError("Unknown patch type %s" % self.patch)

        fieldsplit_0_lu = {
            "ksp_type": "preonly",
            "ksp_max_it": 1,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mkl_pardiso" if self.use_mkl else "mumps",
            "mat_mumps_icntl_14": 200,
        }

        fieldsplit_0_mg = {
            "ksp_type": "richardson",
            "ksp_richardson_self_scale": False,
            "ksp_max_it": 2,
            "ksp_norm_type": "unpreconditioned",
            "ksp_convergence_test": "skip",
            "pc_type": "mg",
            "pc_mg_type": "full",
            "mg_levels": mg_levels_solver,
            "mg_coarse_pc_type": "python",
            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
            "mg_coarse_assembled_pc_type": "lu",
            "mg_coarse_assembled_pc_factor_mat_solver_type": "mkl_pardiso" if self.use_mkl else "superlu_dist"
            # "mg_coarse_assembled_pc_factor_mat_solver_type": "mkl_pardiso" if self.use_mkl else "mumps"
        }
        fieldsplit_0_amg = {
            "ksp_type": "richardson",
            "ksp_max_it": 2,
            "pc_type": "hypre",
        }

        fieldsplit_1 = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "alfi.solver.DGMassInv"
        }

        use_mg = self.solver_type == "almg"

        outer_lu = {
            "mat_type": "aij",
            "ksp_max_it": 1,
            "ksp_convergence_test": "skip",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_14": 200,
            # "mat_mumps_icntl_24": 1,
            # "mat_mumps_icntl_25": 1,
        }

        outer_fieldsplit = {
            "mat_type": "nest",
            "ksp_max_it": 500,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            # "pc_fieldsplit_schur_factorization_type": "upper",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "user",
            "fieldsplit_0": {
                "allu": fieldsplit_0_lu,
                "almg": fieldsplit_0_mg,
                "alamg": fieldsplit_0_amg,
                "lu": None}[self.solver_type],
            "fieldsplit_1": fieldsplit_1,
        }

        outer_base = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_linesearch_maxstep": 1.0,
            "snes_monitor": None,
            "snes_linesearch_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-9,
            "snes_atol": 1.0e-8,
            "snes_stol": 1.0e-6,
            "ksp_type": "fgmres",
            "ksp_rtol": 1.0e-9,
            "ksp_atol": 1.0e-10,
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
        }

        outer = {**outer_base, **outer_lu} if self.solver_type == "lu" else {**outer_base, **outer_fieldsplit}

        parameters["default_sub_matrix_type"] = "aij" if self.use_mkl else "baij"

        if self.tdim > 2:
            outer["ksp_atol"] = 1.0e-8
            outer["ksp_rtol"] = 1.0e-8
            outer["snes_atol"] = outer["ksp_atol"]
            outer["snes_rtol"] = outer["ksp_rtol"]

        return outer

    def message(self, msg):
        if self.mesh.comm.rank == 0:
            warning(msg)


class ConstantPressureSolver(NavierStokesSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def residual(self):
        u, p = split(self.z)
        v, q = TestFunctions(self.Z)
        F = (
            self.nu * inner(2*sym(grad(u)), grad(v))*dx
            + self.gamma * inner(cell_avg(div(u)), div(v))*dx
            + self.advect * inner(dot(grad(u), u), v)*dx
            - p * div(v) * dx
            - div(u) * q * dx
        )
        return F

    def function_space(self, mesh, k):
        tdim = mesh.topological_dimension()
        if k < tdim:
            Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
            FB = FiniteElement("FacetBubble", mesh.ufl_cell(), tdim)
            eleu = VectorElement(NodalEnrichedElement(Pk, FB))
        else:
            Pk = FiniteElement("Lagrange", mesh.ufl_cell(), k)
            eleu = VectorElement(Pk)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, elep)
        return MixedFunctionSpace([V, Q])

    def set_transfers(self):
        V = self.Z.sub(0)
        Q = self.Z.sub(1)
        vtransfer = PkP0SchoeberlTransfer((self.nu, self.gamma), self.tdim, self.hierarchy)
        qtransfer = NullTransfer()
        self.solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=vtransfer.prolong),#, restrict=vtransfer.restrict),
                                           dmhooks.transfer_operators(Q, inject=qtransfer.inject))



class ScottVogeliusSolver(NavierStokesSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def residual(self):
        u, p = split(self.z)
        v, q = TestFunctions(self.Z)
        F = (
            self.nu * inner(2*sym(grad(u)), grad(v))*dx
            + self.gamma * inner(div(u), div(v))*dx
            + self.advect * inner(dot(grad(u), u), v)*dx
            - p * div(v) * dx
            - div(u) * q * dx
        )
        return F

    def function_space(self, mesh, k):
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        V = FunctionSpace(mesh, eleu)
        Q = FunctionSpace(mesh, elep)
        return MixedFunctionSpace([V, Q])

    def set_transfers(self):
        V = self.Z.sub(0)
        Q = self.Z.sub(1)
        if self.stabilisation_type in ["burman", None]:
            qtransfer = NullTransfer()
        elif self.stabilisation_type in ["gls", "supg"]:
            qtransfer = DGTransfer()
        else:
            raise ValueError("Unknown stabilisation")
        transfers = [dmhooks.transfer_operators(Q, inject=qtransfer.inject)]
        if self.hierarchy == "bary":
            vtransfer = SVSchoeberlTransfer((self.nu, self.gamma), self.tdim, self.hierarchy)
            transfers.append(
                dmhooks.transfer_operators(V, prolong=vtransfer.prolong))#, restrict=vtransfer.restrict))
        self.solver.set_transfer_operators(*transfers)
