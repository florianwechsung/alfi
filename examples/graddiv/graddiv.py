from firedrake import *
from alfi.bary import BaryMeshHierarchy
from alfi.transfer import PkP0SchoeberlTransfer, SVSchoeberlTransfer
from alfi import get_default_parser
import argparse
import numpy


parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True, choices=[2, 3])
parser.add_argument("--unstructured", dest="unstructured", default=False, action="store_true")
parser.add_argument("--transfer", dest="transfer", default=False, action="store_true")
parser.add_argument("--monitor", dest="monitor", default=False, action="store_true")
parser.add_argument("--diagonal", type=str, default="left", choices=["left", "right", "crossed"])
parser.add_argument("--mesh", type=str, default=None)
parser.add_argument("--smoother", type=str, default=None, required=True, choices=["patch", "jacobi", "amg"])
args, _ = parser.parse_known_args()

distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

N = args.baseN
nref = args.nref
k = args.k
gamma = Constant(args.gamma)
dim = args.dim
unstructured = args.unstructured


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
    bmesh = Mesh(bary(base._topology_dm), distribution_parameters={"partition": False})
    mh = MeshHierarchy(bmesh, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
elif args.mh == "uniform":
    mh = MeshHierarchy(base, nref, reorder=True, callbacks=(before, after),
                       distribution_parameters=distribution_parameters)
else:
    raise NotImplementedError("Only know bary, uniformbary and uniform for the hierarchy.")

if args.discretisation == "pkp0" and k < dim:
    Pk = FiniteElement("Lagrange", base.ufl_cell(), k)
    FB = FiniteElement("FacetBubble", base.ufl_cell(), dim)
    eleu = VectorElement(NodalEnrichedElement(Pk, FB))
else:
    Pk = FiniteElement("Lagrange", base.ufl_cell(), k)
    eleu = VectorElement(Pk)

V = FunctionSpace(mh[-1], eleu)

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

bcs = DirichletBC(V, 0, "on_boundary")

common = {
    "mat_type": "aij",
    "snes_type": "ksponly",
    "ksp_type": "cg",
    "ksp_rtol": 1e-8,
    "ksp_atol": 0,
    "ksp_max_it": 200,
    # "ksp_monitor_true_residual": None,
    # "ksp_view": None,
    "ksp_norm_type": "unpreconditioned",
}
amg = {
    "pc_type": "hypre"
}
gmg = {
    "pc_type": "mg",
    "pc_mg_cycle_type": "w",
    # "pc_mg_type": "full",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist",
    # "mg_coarse_assembled_pc_factor_mat_solver_type": "petsc",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
}
if args.monitor:
    common["ksp_converged_reason"] = None
    common["ksp_monitor_true_residual"] = None
patch = {
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

if args.patch == "macro":
   patch["mg_levels_patch_pc_patch_construct_type"] = "python"
   patch["mg_levels_patch_pc_patch_construct_python_type"] = "alfi.MacroStar"
   patch["mg_levels_patch_pc_patch_sub_mat_type"] = "aij"
   patch["mg_levels_patch_pc_patch_sub_pc_type"] = "lu"
   patch["mg_levels_patch_sub_pc_factor_mat_solver_type"] = "umfpack"
else:
   patch["mg_levels_patch_pc_patch_construct_type"] = "star"
   patch["mg_levels_patch_pc_patch_construct_dim"] = 0
   patch["mg_levels_patch_pc_patch_sub_mat_type"] = "dense"

pointjacobi = {
    "mg_levels_pc_type": "jacobi",
}
if args.smoother == "patch":
    sp = {**common, **gmg, **patch}
elif args.smoother == "jacobi":
    sp = {**common, **gmg, **pointjacobi}
elif args.smoother == "amg":
    sp = {**common, **amg}
else:
    raise NotImplementedError

pvd = File("output/output.pvd")

if args.discretisation == "sv":
    vtransfer = SVSchoeberlTransfer((1, gamma), args.dim, args.mh)
elif args.discretisation == "pkp0":
    vtransfer = PkP0SchoeberlTransfer((1, gamma), args.dim, args.mh)
nvproblem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver = NonlinearVariationalSolver(nvproblem, solver_parameters=sp, options_prefix="")
if args.transfer:
    transfer = TransferManager(native_transfers={V.ufl_element(): (vtransfer.prolong, vtransfer.restrict, inject)})
    solver.set_transfer_manager(transfer)
gammas = [0, 1, 1e1, 1e2, 1e3, 1e4, 1e6, 1e8]
iters = [">200"] * len(gammas)
for i, gamma_ in enumerate(gammas):
    gamma.assign(gamma_)
    u.assign(0)
    if args.monitor:
        warning("Launching solve for gamma = %s." % gamma_)
    try:
        solver.solve()
    except:
        break

    iters[i] = solver.snes.ksp.getIterationNumber()
    pvd.write(u)
def tostr(i):
    return "%.0e" % i
if base.comm.rank == 0:
    line = ["Ref", "dofs"] + list(map(tostr, gammas))
    print("&" + "\t&\t".join(line) + "\\\\")
    line = map(str, [args.nref, V.dim()] + iters)
    print("&" + "\t&\t".join(line) + "\\\\")
