from firedrake import *
from firedrake.assemble import create_assembly_callable
from firedrake.dmhooks import get_appctx
from firedrake.petsc import *
import weakref
from numpy import unique
import numpy
numpy.set_printoptions(threshold=numpy.inf)
from pyop2.datatypes import IntType
from firedrake.mg.utils import *
from pyop2.profiling import timed_function
from alfi.bubble import BubbleTransfer

def get_basemesh_nodes(W):
    pstart, pend = W.mesh()._topology_dm.getChart()
    section = W.dm.getDefaultSection()
    basemeshoff = {}
    basemeshdof = {}
    degree = W.ufl_element().degree()[1]
    nlayers = W.mesh().layers
    div = nlayers + (degree-1)*(nlayers-1)
    for p in range(pstart, pend):
        dof = section.getDof(p)
        off = section.getOffset(p)
        assert dof%div == 0
        basemeshoff[p] = off
        basemeshdof[p] = dof//div
    return basemeshoff, basemeshdof

@timed_function("fix_coarse_boundaries")
def fix_coarse_boundaries(V):
    if isinstance(V.mesh()._topology, firedrake.mesh.ExtrudedMeshTopology):
        hierarchy, level = get_level(V.mesh())
        dm = V.mesh()._topology_dm
        nlayers = V.mesh().layers
        degree = V.ufl_element().degree()[1]

        baseoff, basedof = get_basemesh_nodes(V)
        section = V.dm.getDefaultSection()
        indices = []
        fStart, fEnd = dm.getHeightStratum(1)
        # Spin over faces, if the face is marked with a magic label
        # value, it means it was in the coarse mesh.
        for p in range(fStart, fEnd):
            value = dm.getLabelValue("prolongation", p)
            if value > -1 and value <= level:
                # OK, so this is a coarse mesh face.
                # Grab all the points in the closure.
                closure, _ = dm.getTransitiveClosure(p)
                for c in closure:
                    # Now add all the dofs on that point to the list
                    # of boundary nodes.
                    dof = basedof[c]
                    off = baseoff[c]
                    for d in range(dof*degree*(nlayers-1)+dof):
                        indices.append(off + d)
        # indices now contains all the nodes on boundaries of course cells on the basemesh extruded up vertically
        # now we need to add every other horizontal layer as well
        for i in range(0, nlayers, 2):
            for p in basedof.keys():
                start = baseoff[p] + degree*i*basedof[p]
                indices += list(range(start, start+basedof[p]))
        nodelist = np.unique(indices).astype(IntType)
    else:
        hierarchy, level = get_level(V.mesh())
        dm = V.mesh()._topology_dm

        section = V.dm.getDefaultSection()
        indices = []
        fStart, fEnd = dm.getHeightStratum(1)
        # Spin over faces, if the face is marked with a magic label
        # value, it means it was in the coarse mesh.
        for p in range(fStart, fEnd):
            value = dm.getLabelValue("prolongation", p)
            if value > -1 and value <= level:
                # OK, so this is a coarse mesh face.
                # Grab all the points in the closure.
                closure, _ = dm.getTransitiveClosure(p)
                for c in closure:
                    # Now add all the dofs on that point to the list
                    # of boundary nodes.
                    dof = section.getDof(c)
                    off = section.getOffset(c)
                    for d in range(dof):
                        indices.append(off + d)
        nodelist = unique(indices).astype(IntType)

    class FixedDirichletBC(DirichletBC):
        def __init__(self, V, g, nodelist):
            self.nodelist = nodelist
            DirichletBC.__init__(self, V, g, "on_boundary")

        @utils.cached_property
        def nodes(self):
            return self.nodelist

    dim = V.mesh().topological_dimension()
    bc = FixedDirichletBC(V, ufl.zero(V.ufl_element().value_shape()), nodelist)
    return bc

class CoarseCellPatches(object):
    def __call__(self, pc):
        from firedrake.mg.utils import get_level
        from firedrake.cython.mgimpl import get_entity_renumbering

        dmf = pc.getDM()
        ctx = pc.getAttr("ctx")

        mf = ctx._x.ufl_domain()
        (mh, level) = get_level(mf)

        coarse_to_fine_cell_map = mh.coarse_to_fine_cells[level-1]
        (_, firedrake_to_plex) = get_entity_renumbering(
            dmf, mf._cell_numbering, "cell")

        patches = []
        for fine_firedrake in coarse_to_fine_cell_map:
            # we need to convert firedrake cell numbering to plex cell numbering
            fine_plex = [firedrake_to_plex[ff] for ff in fine_firedrake]
            entities = []
            for fp in fine_plex:
                (pts, _) = dmf.getTransitiveClosure(fp, True)
                for pt in pts:
                    value = dmf.getLabelValue("prolongation", pt)
                    if not (value > -1 and value <= level):
                        entities.append(pt)

            iset = PETSc.IS().createGeneral(unique(entities),
                                            comm=PETSc.COMM_SELF)
            patches.append(iset)

        piterset = PETSc.IS().createStride(size=len(patches), first=0, step=1,
                                           comm=PETSc.COMM_SELF)
        return (patches, piterset)


class CoarseCellMacroPatches(object):
    def __call__(self, pc):
        from firedrake.mg.utils import get_level
        from firedrake.cython.mgimpl import get_entity_renumbering

        dmf = pc.getDM()
        ctx = pc.getAttr("ctx")

        mf = ctx._x.ufl_domain()
        (mh, level) = get_level(mf)

        coarse_to_fine_cell_map = mh.coarse_to_fine_cells[level-1]
        (_, firedrake_to_plex) = get_entity_renumbering(dmf, mf._cell_numbering, "cell")
        mc = mh[level-1]
        (_, coarse_firedrake_to_plex) = get_entity_renumbering(mc._topology_dm, mc._cell_numbering, "cell")


        patches = []

        tdim = mf.topological_dimension()
        for i, fine_firedrake in enumerate(coarse_to_fine_cell_map):
            # there are d+1 many coarse cells that all map to the same fine cells.
            # We only want to build the patch once, so skip repitions
            if coarse_firedrake_to_plex[i]%(tdim+1) != 0:
                continue
            # we need to convert firedrake cell numbering to plex cell numbering
            fine_plex = [firedrake_to_plex[ff] for ff in fine_firedrake]
            entities = []
            for fp in fine_plex:
                (pts, _) = dmf.getTransitiveClosure(fp, True)
                for pt in pts:
                    value = dmf.getLabelValue("prolongation", pt)
                    if not (value > -1 and value <= level):
                        entities.append(pt)

            iset = PETSc.IS().createGeneral(unique(entities), comm=PETSc.COMM_SELF)
            patches.append(iset)

        piterset = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
        return (patches, piterset)


class ASMPCCoarseCellPatches(ASMPatchPC):

    _prefix = "pc_coarsecell_"

    @timed_function("CoarseCell.get_patches")
    def get_patches(self, V):
        from firedrake.mg.utils import get_level
        from firedrake.cython.mgimpl import get_entity_renumbering

        mesh = V._mesh
        (mh, level) = get_level(mesh)
        mesh_dm = mesh._topology_dm

        section = V.dm.getDefaultSection()

        coarse_to_fine_cell_map = mh.coarse_to_fine_cells[level-1]
        (_, firedrake_to_plex) = get_entity_renumbering(
            mesh_dm, mesh._cell_numbering, "cell")

        patches = []
        points = []
        for fine_firedrake in coarse_to_fine_cell_map:
            # we need to convert firedrake cell numbering to plex cell numbering
            fine_plex = [firedrake_to_plex[ff] for ff in fine_firedrake]
            entities = []
            for fp in fine_plex:
                (pts, _) = mesh_dm.getTransitiveClosure(fp, True)
                for pt in pts:
                    value = mesh_dm.getLabelValue("prolongation", pt)
                    if not (value > -1 and value <= level):
                        entities.append(pt)
            unique_entities = unique(entities)
            points.append(unique_entities)
            indices = []
            for p in unique_entities:
                dof = section.getDof(p)
                if dof <= 0:
                    continue
                off = section.getOffset(p)
                # Local indices within W
                V_indices = numpy.arange(off*V.value_size, V.value_size * (off + dof), dtype='int32')
                indices.extend(V_indices)

            iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            patches.append(iset)
        return patches

class ASMPCHexCoarseCellPatches(ASMPatchPC):

    _prefix = "pc_coarsecell_"

    @timed_function("HexCoarseCell.get_patches")
    def get_patches(self, V):
        mesh = V._mesh
        (mh, level) = get_level(mesh)
        nlayers = mesh.layers
        mesh_dm = mesh._topology_dm


        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = []
        for (i, W) in enumerate(V):
            V_local_ises_indices.append(V.dof_dset.local_ises[i].indices)

        
        basemeshoff = []
        basemeshdof = []
        dm = mesh._topology_dm

        for (i, W) in enumerate(V):
            boff, bdof = get_basemesh_nodes(W)
            basemeshoff.append(boff)
            basemeshdof.append(bdof)

        # Build index sets for the patches
        ises = []
        (start, end) = mesh_dm.getDepthStratum(0)
        for seed in range(start, end):
            # Only build patches over owned DoFs
            if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
                continue
            value = mesh_dm.getLabelValue("prolongation", seed)
            if (value > -1 and value <= level):
                continue

            # Create point list from mesh DM
            pt_array, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
            for k in range(1, nlayers, 2):
                indices = []
                # Get DoF indices for patch
                for (i, W) in enumerate(V):
                    section = W.dm.getDefaultSection()
                    degree = W.ufl_element().degree()[1]
                    for p in pt_array.tolist():
                        dof = basemeshdof[i][p]
                        if dof <= 0:
                            continue
                        off = basemeshoff[i][p]
                        if k == 0:
                            begin = off
                            end = off+dof+(degree-1)*dof
                        elif k == nlayers-1:
                            begin = off+k*degree*dof-(degree-1)*dof
                            end = off+dof+k*degree*dof
                        else:
                            begin = off+k*degree*dof-(degree-1)*dof
                            end = off+dof+k*degree*dof+(degree-1)*dof
                        W_indices = np.arange(W.value_size*begin, W.value_size*end, dtype='int32')
                        indices.extend(V_local_ises_indices[i][W_indices])
                iset = PETSc.IS().createGeneral(indices, comm=COMM_SELF)
                ises.append(iset)
        return ises


class AutoSchoeberlTransfer(object):
    def __init__(self, parameters, tdim, hierarchy, backend='pcpatch', b_matfree=True, hexmesh=False):
        assert backend in ['pcpatch', 'tinyasm', 'petscasm', 'lu']
        self.solver = {}
        self.bcs = {}
        self.rhs = {}
        self.tensors = {}
        self.parameters = parameters
        self.prev_parameters = {}
        self.force_rebuild_d = {}
        patchparams = {"snes_type": "ksponly",
                       "ksp_type": "preonly",
                       "ksp_convergence_test": "skip",
                       }
        if backend == 'pcpatch':
            backendparams = {
                "mat_type": "matfree",
                "pc_type": "python",
                "pc_python_type": "firedrake.PatchPC",
                "patch_pc_patch_save_operators": "true",
                "patch_pc_patch_partition_of_unity": False,
                "patch_pc_patch_multiplicative": False,
                "patch_pc_patch_sub_mat_type": "seqaij" if tdim > 2 else "seqdense",
                "patch_pc_patch_construct_type": "python",
                "patch_pc_patch_construct_python_type": "alfi.transfer.CoarseCellMacroPatches" if hierarchy == "bary" else "alfi.transfer.CoarseCellPatches",
                "patch_sub_ksp_type": "preonly",
                "patch_sub_pc_type": "lu"
            }
        elif backend == 'tinyasm' or backend == 'petscasm':
            if hierarchy == 'bary':
                raise NotImplementedError('ASMPCCoarseCellMacroPatches not yet implemented')

            backendparams = {
                "mat_type": "aij",
                "pc_type": "python",
                "pc_python_type": "alfi.transfer.ASMPCHexCoarseCellPatches" if hexmesh else "alfi.transfer.ASMPCCoarseCellPatches",
                "pc_coarsecell_backend": backend,
            }
        else: 
            backendparams = {
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        self.patchparams = {**patchparams, **backendparams}
        self.b_matfree = b_matfree

    def break_ref_cycles(self):
        for attr in ["solver", "bcs", "rhs", "tensors", "parameters", "prev_parameters"]:
            if hasattr(self, attr):
                delattr(self, attr)


    def bform(self, rhs):
        a = get_appctx(rhs.function_space().dm).J
        return action(a, rhs)

    def form(self, V):
        a = get_appctx(V.dm).J
        return a

    def force_rebuild(self):
        self.force_rebuild_d = {}
        for k in self.prev_parameters:
            self.force_rebuild_d[k] = True

    def rebuild(self, key):
        if key in self.force_rebuild_d and self.force_rebuild_d[key]:
            self.force_rebuild_d[key] = False
            warning(RED % ("Rebuild prolongation for key %i" % key))
            return True
        prev_parameters = self.prev_parameters.get(key, [])
        update = False
        for (prev_param, param) in zip(prev_parameters, self.parameters):
            #if float(param) != prev_param:
            if param != prev_param:
                update = True
                break
        return update

    @timed_function("SchoeberlProlong")
    def prolong(self, coarse, fine):
        self.restrict_or_prolong(coarse, fine, "prolong")

    @timed_function("SchoeberlRestrict")
    def restrict(self, fine, coarse):
        self.restrict_or_prolong(fine, coarse, "restrict")

    def restrict_or_prolong(self, source, target, mode):
        if mode == "prolong":
            coarse = source
            fine = target
        else:
            fine = source
            coarse = target
        # Rebuild without any indices
        V = FunctionSpace(fine.ufl_domain(), fine.function_space().ufl_element())
        key = V.dim()

        firsttime = self.bcs.get(key, None) is None

        if firsttime:
            from firedrake.solving_utils import _SNESContext
            bcs = fix_coarse_boundaries(V)
            a = self.form(V)
            A = assemble(a, bcs=bcs, mat_type=self.patchparams["mat_type"])


            tildeu, rhs = Function(V), Function(V)

            if self.b_matfree:
                bform = self.bform(rhs)
            else:
                bform = assemble(self.bform(TrialFunction(V)))
            b = Function(V)
            problem = LinearVariationalProblem(a=a, L=0, u=tildeu, bcs=bcs)
            ctx = _SNESContext(problem, mat_type=self.patchparams["mat_type"],
                               pmat_type=self.patchparams["mat_type"],
                               appctx={}, options_prefix="prolongation")

            solver = LinearSolver(A, solver_parameters=self.patchparams,
                                  options_prefix="prolongation")
            solver._ctx = ctx
            self.bcs[key] = bcs
            self.solver[key] = solver
            self.rhs[key] = tildeu, rhs
            self.tensors[key] = A, b, bform
            #self.prev_parameters[key] = [float(param) for param in self.parameters]
            self.prev_parameters[key] = self.parameters
        else:
            bcs = self.bcs[key]
            solver = self.solver[key]
            A, b, bform = self.tensors[key]
            tildeu, rhs = self.rhs[key]

            # Update operator if parameters have changed.

            if self.rebuild(key):
                A = solver.A
                a = self.form(V)
                if self.b_matfree:
                    bform = self.bform(rhs)
                else:
                    bform = assemble(self.bform(TrialFunction(V)))
                self.tensors[key] = A, b, bform
                A = assemble(a, bcs=bcs, mat_type=self.patchparams["mat_type"], tensor=A)
                #self.prev_parameters[key] = [float(param) for param in self.parameters]
                self.prev_parameters[key] = self.parameters

        if mode == "prolong":
            self.standard_transfer(coarse, rhs, "prolong")

            if self.b_matfree:
                b = assemble(bform, bcs=bcs, tensor=b)
            else:
                with rhs.dat.vec_ro as rhsv:
                    with b.dat.vec_wo as out:
                        bform.petscmat.mult(rhsv, out)
                bcs.apply(b)
            # # Could do
            # # solver.solve(tildeu, b)
            # # but that calls a lot of SNES and KSP overhead.
            # # We know we just want to apply the PC:
            with solver.inserted_options(), dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with b.dat.vec_ro as rhsv:
                    with tildeu.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            # fine.assign(rhs - tildeu)
            fine.dat.data[:] = rhs.dat.data_ro - tildeu.dat.data_ro

        else:
            # restrict(rhs, coarse)
            # return
            # tildeu.assign(fine)
            tildeu.dat.data[:] = fine.dat.data_ro
            bcs.apply(tildeu)
            with solver.inserted_options(), dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with tildeu.dat.vec_ro as rhsv:
                    with rhs.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            # solver.solve(rhs, fine)
            if self.b_matfree:
                b = assemble(bform, tensor=b)
            else:
                with rhs.dat.vec_ro as rhsv:
                    with b.dat.vec_wo as out:
                        bform.petscmat.mult(rhsv, out)
            # rhs.assign(fine-b)
            rhs.dat.data[:] = fine.dat.data_ro - b.dat.data_ro
            self.standard_transfer(rhs, coarse, "restrict")


        # def energy_norm(u):
        #     return assemble(action(action(self.form(u.function_space()), u), u))
        # if mode == "prolong":
        #     warning("From mesh %i to %i" % (coarse.function_space().dim(), fine.function_space().dim()))
        #     warning("Energy norm ratio from %.2f to %.2f" % ((energy_norm(rhs)/energy_norm(coarse)), energy_norm(fine)/energy_norm(coarse)))

    @timed_function('standard_transfer')
    def standard_transfer(self, source, target, mode):
        if mode == "prolong":
            prolong(source, target)
        elif mode == "restrict":
            restrict(source, target)
        else:
            raise NotImplementedError


class SVSchoeberlTransfer(AutoSchoeberlTransfer):

    def form(self, V):
        (nu, gamma) = self.parameters
        u = TrialFunction(V)
        v = TestFunction(V)
        a = nu * inner(2*sym(grad(u)), grad(v))*dx + gamma*inner(div(u), div(v))*dx
        return a

    def bform(self, rhs):
        V = rhs.function_space()
        (nu, gamma) = self.parameters
        u = TrialFunction(V)
        v = TestFunction(V)
        a = gamma*inner(div(u), div(v))*dx
        # a = nu * inner(2*sym(grad(u)), grad(v))*dx + gamma*inner(div(u), div(v))*dx
        return action(a, rhs)


class PkP0SchoeberlTransfer(AutoSchoeberlTransfer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transfers = {}


    def form(self, V):
        (nu, gamma) = self.parameters
        appctx = get_appctx(V.dm).appctx
        if 'deg' in appctx:
            deg = appctx['deg']
        else:
            deg = None
        u = TrialFunction(V)
        v = TestFunction(V)
        if (callable(nu)):
            (mh, level) = get_level(u.ufl_domain())
            a = nu(mh[level]) * inner(2*sym(grad(u)), grad(v))*dx(degree=deg) + \
                gamma*inner(cell_avg(div(u)), div(v))*dx(metadata={"mode": "vanilla"})
        else:
            a = nu * inner(2*sym(grad(u)), grad(v))*dx(degree=deg) + \
                gamma*inner(cell_avg(div(u)), div(v))*dx(metadata={"mode": "vanilla"})

        return a

    def bform(self, rhs):
        V = rhs.function_space()
        (nu, gamma) = self.parameters
        u = TrialFunction(V)
        v = TestFunction(V)
        a = gamma*inner(cell_avg(div(u)), div(v))*dx(metadata={"mode": "vanilla"})
        return action(a, rhs)

    def standard_transfer(self, source, target, mode):
        if not (source.ufl_shape[0] == 3 and
                "CG1" in source.ufl_element().shortstr()):
            return super().standard_transfer(source, target, mode)

        if mode == "prolong":
            coarse = source
            fine = target
        elif mode == "restrict":
            fine = source
            coarse = target
        else:
            raise NotImplementedError
        (mh, level) = get_level(coarse.ufl_domain())
        if level not in self.transfers:
            self.transfers[level] = BubbleTransfer(
                coarse.function_space(), fine.function_space())
        if mode == "prolong":
            self.transfers[level].prolong(coarse, fine)
        elif mode == "restrict":
            self.transfers[level].restrict(fine, coarse)
        else:
            raise NotImplementedError

class AlgebraicSchoeberlTransfer(object):
    def __init__(self, parameters, A_callback, BTWB_callback, tdim, hierarchy, backend='tinyasm', hexmesh=False):
        assert backend in ['tinyasm', 'petscasm', 'lu']
        assert hierarchy == 'uniform'
        self.tdim = tdim
        self.solver = {}
        self.bcs = {}
        self.rhs = {}
        self.tensors = {}
        self.parameters = parameters
        self.prev_parameters = {}
        self.force_rebuild_d = {}
        patchparams = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "ksp_convergence_test": "skip",
            "mat_type": "aij",
        }
        if backend in ['tinyasm', 'petscasm']:
            backendparams = {
                "pc_type": "python",
                "pc_python_type": "alfi.transfer.ASMPCHexCoarseCellPatches" if hexmesh else "alfi.transfer.ASMPCCoarseCellPatches",
                "pc_coarsecell_backend": backend,
                # "pc_coarsecell_sub_pc_asm_sub_mat_type": "seqaij",
                # "pc_coarsecell_sub_sub_pc_factor_mat_solver_type": "umfpack",
            }
        else:
            backendparams = {
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        self.patchparams = {**patchparams, **backendparams}

        self.A_callback = A_callback
        self.BTWB_callback = BTWB_callback

    def break_ref_cycles(self):
        for attr in ["solver", "bcs", "rhs", "tensors", "parameters", "prev_parameters"]:
            if hasattr(self, attr):
                delattr(self, attr)


    def force_rebuild(self):
        self.force_rebuild_d = {}
        for k in self.prev_parameters:
            self.force_rebuild_d[k] = True

    def rebuild(self, key):
        if key in self.force_rebuild_d and self.force_rebuild_d[key]:
            self.force_rebuild_d[key] = False
            warning(RED % ("Rebuild prolongation for key %i" % key))
            return True
        prev_parameters = self.prev_parameters.get(key, [])
        update = False
        for (prev_param, param) in zip(prev_parameters, self.parameters):
            #if float(param) != prev_param:
            if param != prev_param:
                update = True
                break
        return update

    @timed_function("SchoeberlProlong")
    def prolong(self, coarse, fine):
        self.restrict_or_prolong(coarse, fine, "prolong")

    @timed_function("SchoeberlRestrict")
    def restrict(self, fine, coarse):
        self.restrict_or_prolong(fine, coarse, "restrict")

    def restrict_or_prolong(self, source, target, mode):
        if mode == "prolong":
            coarse = source
            fine = target
        else:
            fine = source
            coarse = target
        # Rebuild without any indices
        V = FunctionSpace(fine.ufl_domain(), fine.function_space().ufl_element())
        key = V.dim()

        firsttime = self.bcs.get(key, None) is None

        _, level = get_level(V.ufl_domain())
        if firsttime:
            from firedrake.solving_utils import _SNESContext
            bcs = fix_coarse_boundaries(V)
            A = self.A_callback(level, mat=None)
            BTWB = self.BTWB_callback(level, mat=None)

            rmap, cmap = A.petscmat.getLGMap()
            A.petscmat.axpy(1, BTWB, A.petscmat.Structure.SUBSET_NONZERO_PATTERN)
            A.petscmat.setLGMap(rmap, cmap)
            for i in range(self.tdim):
                A.petscmat.zeroRowsLocal(i+bcs.nodes*self.tdim, 1.0)

            tildeu, rhs = Function(V), Function(V)

            b = Function(V)
            problem = LinearVariationalProblem(a=A.form, L=0, u=tildeu, bcs=bcs)
            ctx = _SNESContext(problem, mat_type=self.patchparams["mat_type"],
                               pmat_type=self.patchparams["mat_type"],
                               appctx={}, options_prefix="prolongation")

            solver = LinearSolver(A, solver_parameters=self.patchparams,
                                  options_prefix="prolongation")
            solver._ctx = ctx
            self.bcs[key] = bcs
            self.solver[key] = solver
            self.rhs[key] = tildeu, rhs
            # self.tensors[key] = A, b, bform
            self.tensors[key] = A, b, BTWB
            #self.prev_parameters[key] = [float(param) for param in self.parameters]
            self.prev_parameters[key] = self.parameters
        else:
            bcs = self.bcs[key]
            solver = self.solver[key]
            # A, b, bform = self.tensors[key]
            A, b, BTWB = self.tensors[key]
            tildeu, rhs = self.rhs[key]

            # Update operator if parameters have changed.

            if self.rebuild(key):
                A = solver.A
                self.A_callback(level, mat=A)
                self.BTWB_callback(level, mat=BTWB)
                A.petscmat += BTWB
                for i in range(self.tdim):
                    A.petscmat.zeroRowsLocal(i+bcs.nodes*self.tdim, 1.0)
                self.tensors[key] = A, b, BTWB
                # A = assemble(a, bcs=bcs, mat_type=self.patchparams["mat_type"], tensor=A)
                #self.prev_parameters[key] = [float(param) for param in self.parameters]
                self.prev_parameters[key] = self.parameters

        if mode == "prolong":
            self.standard_transfer(coarse, rhs, "prolong")

            with rhs.dat.vec_ro as rhsvec:
                with b.dat.vec_wo as bvec:
                    BTWB.mult(rhsvec, bvec)
            bcs.apply(b)
            with solver.inserted_options(), dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with b.dat.vec_ro as rhsv:
                    with tildeu.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            fine.dat.data[:] = rhs.dat.data_ro - tildeu.dat.data_ro

        else:
            tildeu.dat.data[:] = fine.dat.data_ro
            bcs.apply(tildeu)
            with solver.inserted_options(), dmhooks.add_hooks(solver.ksp.dm, solver, appctx=solver._ctx):
                with tildeu.dat.vec_ro as rhsv:
                    with rhs.dat.vec_wo as x:
                        solver.ksp.pc.apply(rhsv, x)
            with rhs.dat.vec_ro as rhsvec:
                with b.dat.vec_wo as bvec:
                    BTWB.mult(rhsvec, bvec)
            rhs.dat.data[:] = fine.dat.data_ro - b.dat.data_ro
            self.standard_transfer(rhs, coarse, "restrict")


        def h1_norm(u):
            return assemble(inner(grad(u), grad(u)) * dx)

        def energy_norm(u):
            v = u.copy(deepcopy=True)
            with u.dat.vec_ro as x:
                with v.dat.vec_wo as y:
                    A.petscmat.mult(x, y)
                    res = x.dot(y)
            return res

        def divergence_norm(u):
            v = u.copy(deepcopy=True)
            with u.dat.vec_ro as x:
                with v.dat.vec_wo as y:
                    BTWB.mult(x, y)
                    res = x.dot(y)
            return res

        # if mode == "prolong":
        #     warning("From mesh %i to %i" % (coarse.function_space().dim(), fine.function_space().dim()))
        #     warning("Energy norm from %.4f to %.4f" % (energy_norm(rhs), energy_norm(fine)))
        #     warning("H1 norm from %.4f to %.4f" % (h1_norm(rhs), h1_norm(fine)))
        #     warning("Divergence norm from %.4f to %.4f" % (divergence_norm(rhs), divergence_norm(fine)))
        #     File('/tmp/tmp.pvd').write(b)
        #     import sys; sys.exit()

    def standard_transfer(self, source, target, mode):
        if mode == "prolong":
            prolong(source, target)
        elif mode == "restrict":
            restrict(source, target)
        else:
            raise NotImplementedError

class NullTransfer(object):
    def transfer(self, src, dest):
        with dest.dat.vec_wo as x:
            x.set(numpy.nan)

    inject = transfer
    prolong = transfer
    restrict = transfer


class DGInjection(object):

    def __init__(self):
        self._DG_inv_mass = {}
        self._mixed_mass = {}
        self._tmp_function = {}

    def DG_inv_mass(self, DG):
        """
        Inverse DG mass matrix
        :arg DG: the DG space
        :returns: A PETSc Mat.
        """
        key = DG.dim()
        try:
            return self._DG_inv_mass[key]
        except KeyError:
            assert DG.ufl_element().family() == "Discontinuous Lagrange"
            M = assemble(Tensor(inner(TestFunction(DG), TrialFunction(DG))*dx).inv)
            return self._DG_inv_mass.setdefault(key, M.petscmat)

    def mixed_mass(self, V_A, V_B):
        """
        Compute the mixed mass matrix of two function spaces.
        :arg V_A: the donor space
        :arg V_B: the target space
        :returns: A PETSc Mat.
        """
        from firedrake.supermeshing import assemble_mixed_mass_matrix
        key = (V_A.dim(), V_B.dim())
        try:
            return self._mixed_mass[key]
        except KeyError:
            M = assemble_mixed_mass_matrix(V_A, V_B)
            return self._mixed_mass.setdefault(key, M)

    def tmp_function(self, V):
        """
        Construct a temporary work function on a function space.
        """
        key = V.dim()
        try:
            return self._tmp_function[key]
        except KeyError:
            u = Function(V)
            return self._tmp_function.setdefault(key, u)

    def inject(self, fine, coarse):
        V_fine = fine.function_space()
        V_coarse = coarse.function_space()

        mixed_mass = self.mixed_mass(V_fine, V_coarse)
        mass_inv = self.DG_inv_mass(V_coarse)
        tmp = self.tmp_function(V_coarse)

        # Can these be bundled into one with statement?
        with fine.dat.vec_ro as src, tmp.dat.vec_wo as rhs:
            mixed_mass.mult(src, rhs)
        with tmp.dat.vec_ro as rhs, coarse.dat.vec_wo as dest:
            mass_inv.mult(rhs, dest)

    prolong = inject
