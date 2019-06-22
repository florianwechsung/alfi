from firedrake import *
from alfi.bary import BaryMeshHierarchy
from alfi.transfer import PkP0SchoeberlTransfer, SVSchoeberlTransfer
from alfi import get_default_parser
import argparse
import numpy


parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True, choices=[2, 3])
parser.add_argument("--unstructured", dest="unstructured", default=False, action="store_true")
parser.add_argument("--diagonal", type=str, default="left", choices=["left", "right", "crossed"])
parser.add_argument("--mesh", type=str, default=None)
args, _ = parser.parse_known_args()

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

N = args.baseN
nref = args.nref
k = args.k
gamma = Constant(args.gamma)
dim = args.dim
unstructured = args.unstructured

warning("Let's make some meshes.")

if args.mesh is not None:
    warning("Ignoring --baseN, using unstructured Gmsh meshes")
    base = Mesh(args.mesh)
else:
    if dim == 2:
        base = UnitSquareMesh(N, N, distribution_parameters=distribution_parameters, diagonal=args.diagonal)
    elif dim == 3:
        base = UnitCubeMesh(N, N, N, distribution_parameters=distribution_parameters)

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

if args.mh == "bary":
    mh = BaryMeshHierarchy(base, nref, callbacks=(before, after),
                           reorder=True, distribution_parameters=distribution_parameters)
elif args.mh == "uniformbary":
    bmesh = Mesh(bary(base._plex), distribution_parameters={"partition": False})
    mh = MeshHierarchy(bmesh, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
elif args.mh == "uniform":
    mh = MeshHierarchy(base, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
else:
    raise NotImplementedError("Only know bary, uniformbary and uniform for the hierarchy.")
warning("Meshes are ready.")

if args.discretisation == "pkp0" and k < dim:
    Pk = FiniteElement("Lagrange", base.ufl_cell(), k)
    FB = FiniteElement("FacetBubble", base.ufl_cell(), dim)
    eleu = VectorElement(NodalEnrichedElement(Pk, FB))
else:
    Pk = FiniteElement("Lagrange", base.ufl_cell(), k)
    eleu = VectorElement(Pk)

# Pk = FiniteElement("Lagrange", base.ufl_cell(), 3)
# FB = FiniteElement("FacetBubble", base.ufl_cell(), dim)
# eleu = VectorElement(NodalEnrichedElement(Pk, FB))
# eleu = VectorElement(Pk)
V = FunctionSpace(mh[-1], eleu)
warning("Function space ready.")

u = Function(V, name="Solution")
v = TestFunction(V)

if dim == 2:
    f = Constant((1, 1))
    bclabels = [1, 2, 3, 4]
else:
    f = Constant((1, 1, 1))
    bclabels = [1, 2, 3, 4, 5, 6]
if args.discretisation == "pkp0":
    F = inner(2*sym(grad(u)), grad(v))*dx + gamma*inner(cell_avg(div(u)), div(v))*dx - inner(f, v)*dx
else:
    F = inner(2*sym(grad(u)), grad(v))*dx + gamma*inner(div(u), div(v))*dx - inner(f, v)*dx

warning("Creating Dirichlet boundary condition.")
bcs = DirichletBC(V, 0, "on_boundary")
warning("Boundary condition ready.")

sp = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    # "ksp_type": "fgmres",
    "ksp_type": "fcg",
    "ksp_rtol": 1e-8,
    "ksp_atol": 0,
    "ksp_max_it": 1000,
    # "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    # "ksp_view": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_type": "mg",
    "pc_mg_cycle_type": "w",
    # "pc_mg_type": "full",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    # "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "petsc",
    # "mg_levels_ksp_type": "richardson",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_pc_side": "right", # for some reason, gmres with left preconditioning does quite poorly
    "mg_levels_ksp_max_it": 6 if dim == 3 else 2,
    "mg_levels_ksp_richardson_scale": 1/(dim+1),
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_sub_mat_type": "aij",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}
if args.patch == "macro":
   sp["mg_levels_patch_pc_patch_construct_type"] = "python"
   sp["mg_levels_patch_pc_patch_construct_python_type"] = "alfi.MacroStar"
else:
   sp["mg_levels_patch_pc_patch_construct_type"] = "star"
   sp["mg_levels_patch_pc_patch_construct_dim"] = 0

pvd = File("output/output.pvd")

iters = []
gammas = [0, 1, 1e1, 1e2, 1e4, 1e6, 1e8]
# gammas = [1e2, 1e4, 1e6, 1e8]
for gamma_ in gammas:
    if args.discretisation == "sv":
        vtransfer = SVSchoeberlTransfer((1, gamma), args.dim, args.mh)
    elif args.discretisation == "pkp0":
        vtransfer = PkP0SchoeberlTransfer((1, gamma), args.dim, args.mh)
    gamma.assign(gamma_)
    u.assign(0)
    warning("Launching solve for gamma = %s." % gamma_)
    nvproblem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(nvproblem, solver_parameters=sp, options_prefix="")
    solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=vtransfer.prolong, restrict=vtransfer.restrict))
    # solver.set_transfer_operators(dmhooks.transfer_operators(V, restrict=vtransfer.restrict))
    # solver.set_transfer_operators(dmhooks.transfer_operators(V, prolong=vtransfer.prolong))
    solver.solve()
    iters.append(solver.snes.ksp.getIterationNumber())
    pvd.write(u)
def tostr(i):
    return "%.0e" % i
line = ["Ref", "dofs"] + list(map(tostr, gammas))
print("&" + "\t&\t".join(line) + "\\\\")
line = map(str, [args.nref, V.dim()] + iters)
print("&" + "\t&\t".join(line) + "\\\\")
