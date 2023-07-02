from firedrake import *
import firedrake
from pyop2.datatypes import IntType
from mpi4py import MPI
from fractions import Fraction
from firedrake.cython import mgimpl as impl
from firedrake.cython.dmcommon import FACE_SETS_LABEL
from firedrake.petsc import *
import numpy

__all__ = ["BaryMeshHierarchy"]

def closure(dm, p):
    return dm.getTransitiveClosure(p, useCone=True)

def bary(cdm):

    for v in range(*cdm.getDepthStratum(0)):
        cdm.setLabelValue("MacroVertices", v, 1)

    tr = PETSc.DMPlexTransform().create(comm=cdm.getComm())
    tr.setType(PETSc.DMPlexTransformType.REFINEALFELD)
    tr.setDM(cdm)
    tr.setUp()
    rdm = tr.apply(cdm)

    return rdm

def BaryMeshHierarchy(mesh, refinement_levels, distribution_parameters=None, callbacks=None, reorder=None,
                      refinements_per_level=1):
    cdm = mesh._topology_dm
    cdm.setRefinementUniform(True)
    dms = []
    if mesh.comm.size > 1 and mesh._grown_halos:
        raise RuntimeError("Cannot refine parallel overlapped meshes "
                           "(make sure the MeshHierarchy is built immediately after the Mesh)")
    parameters = {}
    if distribution_parameters is not None:
        parameters.update(distribution_parameters)
    else:
        parameters.update(mesh._distribution_parameters)

    parameters["partition"] = False
    distribution_parameters = parameters

    if callbacks is not None:
        before, after = callbacks
    else:
        before = after = lambda dm, i: None

    for i in range(refinement_levels*refinements_per_level):
        if i % refinements_per_level == 0:
            before(cdm, i)
        rdm = cdm.refine()
        if i % refinements_per_level == 0:
            after(rdm, i)
        # Remove interior facet label (re-construct from
        # complement of exterior facets).  Necessary because the
        # refinement just marks points "underneath" the refined
        # facet with the appropriate label.  This works for
        # exterior, but not marked interior facets
        rdm.removeLabel("interior_facets")
        # Remove vertex (and edge) points from labels on exterior
        # facets.  Interior facets will be relabeled in Mesh
        # construction below.
        impl.filter_labels(rdm, rdm.getHeightStratum(1),
                           "exterior_facets", "boundary_faces",
                           FACE_SETS_LABEL)

        rdm.removeLabel("pyop2_core")
        rdm.removeLabel("pyop2_owned")
        rdm.removeLabel("pyop2_ghost")

        dms.append(rdm)
        cdm = rdm
        # Fix up coords if refining embedded circle or sphere
        if hasattr(mesh, '_radius'):
            # FIXME, really we need some CAD-like representation
            # of the boundary we're trying to conform to.  This
            # doesn't DTRT really for cubed sphere meshes (the
            # refined meshes are no longer gnonomic).
            coords = cdm.getCoordinatesLocal().array.reshape(-1, mesh.geometric_dimension())
            scale = mesh._radius / np.linalg.norm(coords, axis=1).reshape(-1, 1)
            coords *= scale

    barydms = (bary(mesh._topology_dm), ) + tuple(bary(dm) for dm in dms)

    for bdm in barydms:
        impl.filter_labels(bdm, bdm.getHeightStratum(1),
                           "exterior_facets", "boundary_faces",
                           FACE_SETS_LABEL)

    barymeshes = [firedrake.Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                 distribution_parameters=distribution_parameters,
                                 comm=mesh.comm,
                                 reorder=reorder)
                           for dm in barydms]

    meshes = [mesh] + [firedrake.Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(),
                                      distribution_parameters=distribution_parameters,
                                      comm=mesh.comm,
                                      reorder=reorder)
                       for dm in dms]

    lgmaps = []
    for i, m in enumerate(meshes):
        no = impl.create_lgmap(m._topology_dm)
        m.init()
        o = impl.create_lgmap(m._topology_dm)
        m._topology_dm.setRefineLevel(i)
        lgmaps.append((no, o))

    coarse_to_fine_cells = []
    fine_to_coarse_cells = [None]
    for (coarse, fine), (clgmaps, flgmaps) in zip(zip(meshes[:-1], meshes[1:]),
                                                  zip(lgmaps[:-1], lgmaps[1:])):
        c2f, f2c = impl.coarse_to_fine_cells(coarse, fine, clgmaps, flgmaps)
        coarse_to_fine_cells.append(c2f)
        fine_to_coarse_cells.append(f2c)

    lgmaps = []
    for i, m in enumerate(barymeshes):
        no = impl.create_lgmap(m._topology_dm)
        m.init()
        o = impl.create_lgmap(m._topology_dm)
        m._topology_dm.setRefineLevel(i)
        lgmaps.append((no, o))

    d = mesh.topological_dimension()
    bary_coarse_to_fine_cells = []
    bary_fine_to_coarse_cells = [None]
    for (coarseu, fineu), (coarse, fine), (clgmaps, flgmaps), uniform_coarse_to_fine \
        in zip(zip(meshes[:-1], meshes[1:]),
               zip(barymeshes[:-1], barymeshes[1:]),
               zip(lgmaps[:-1], lgmaps[1:]),
               coarse_to_fine_cells):

        cdm = coarseu._topology_dm
        fdm = fineu._topology_dm
        _, cn2o = impl.get_entity_renumbering(cdm, coarseu._cell_numbering, "cell")
        _, fn2o = impl.get_entity_renumbering(fdm, fineu._cell_numbering, "cell")
        plex_uniform_coarse_to_fine = numpy.empty_like(uniform_coarse_to_fine)
        for i, cells in enumerate(uniform_coarse_to_fine):
            plexcells = fn2o[cells]
            plex_uniform_coarse_to_fine[cn2o[i], :] = plexcells

        ncoarse, nfine = plex_uniform_coarse_to_fine.shape
        plex_coarse_bary_to_fine_bary = numpy.full((ncoarse*(d+1),
                                                    nfine*(d+1)), -1, dtype=PETSc.IntType)

        for c in range(ncoarse*(d+1)):
            uniform = c // (d+1)
            fine_cells = plex_uniform_coarse_to_fine[uniform]
            bary_cells = []
            for fc in fine_cells:
                bary_cells.extend(list(range(fc*(d+1), (fc+1)*(d+1))))
            plex_coarse_bary_to_fine_bary[c] = bary_cells

        cdm = coarse._topology_dm
        fdm = fine._topology_dm

        co2n, _ = impl.get_entity_renumbering(cdm, coarse._cell_numbering, "cell")
        fo2n, _ = impl.get_entity_renumbering(fdm, fine._cell_numbering, "cell")

        coarse_bary_to_fine_bary = numpy.empty_like(plex_coarse_bary_to_fine_bary)
        # Translate plex numbering to firedrake numbering
        for i, plex_cells in enumerate(plex_coarse_bary_to_fine_bary):
            coarse_bary_to_fine_bary[co2n[i]] = fo2n[plex_cells]

        bary_coarse_to_fine_cells.append(coarse_bary_to_fine_bary)

        # Not fast but seems to work
        fine_bary_to_coarse_bary = [[]]
        for i in range(numpy.max(coarse_bary_to_fine_bary)):
            fine_bary_to_coarse_bary.append([])
        for coarse in range(coarse_bary_to_fine_bary.shape[0]):
            for ifine in range(coarse_bary_to_fine_bary.shape[1]):
                # the coarse cell `coarse` is contained in the fine cell
                # `bary_coarse_to_fine_cells[0][coarse, ifine]` so we
                # should add it to the corresponding list
                fine_bary_to_coarse_bary[coarse_bary_to_fine_bary[coarse, ifine]].append(coarse)
        fine_bary_to_coarse_bary = numpy.asarray(fine_bary_to_coarse_bary, dtype=PETSc.IntType)

        bary_fine_to_coarse_cells.append(fine_bary_to_coarse_bary)

    #print(bary_coarse_to_fine_cells)
    #print(bary_fine_to_coarse_cells)

    coarse_to_fine_cells = dict((Fraction(i, refinements_per_level), c2f)
                                for i, c2f in enumerate(bary_coarse_to_fine_cells))
    fine_to_coarse_cells = dict((Fraction(i, refinements_per_level), f2c)
                                for i, f2c in enumerate(bary_fine_to_coarse_cells))
    return HierarchyBase(barymeshes, coarse_to_fine_cells, fine_to_coarse_cells,
                         refinements_per_level, nested=False)
