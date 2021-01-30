from firedrake import *
from firedrake.petsc import *
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


def select_entity(p, dm=None, exclude=None):
    """Filter entities based on some label.

    :arg p: the entity.
    :arg dm: the DMPlex object to query for labels.
    :arg exclude: The label marking points to exclude."""
    if exclude is None:
        return True
    else:
        # If the exclude label marks this point (the value is not -1),
        # we don't want it.
        return dm.getLabelValue(exclude, p) == -1


class OrderedRelaxation(object):
    def __init__(self):
        self.name = None

    def callback(self, dm, entity):
        raise NotImplementedError

    def set_options(self, dm, opts, name):
        pass

    @staticmethod
    def star(dm, p):
        return dm.getTransitiveClosure(p, useCone=False)[0]

    @staticmethod
    def closure(dm, p):
        return dm.getTransitiveClosure(p, useCone=True)[0]

    @staticmethod
    def cone(dm, p):
        return dm.getCone(p)

    @staticmethod
    def support(dm, p):
        return dm.getSupport(p)

    @staticmethod
    def exterior_facets(dm):
        return dm.getStratumIS("exterior_facets", 1).getIndices()

    @staticmethod
    def exterior_vertices(dm):
        (vlo, vhi) = dm.getDepthStratum(0)
        exterior_facets = PatchConstructor.exterior_facets(dm)

        keep = lambda e: vlo <= e < vhi
        exterior_vertices = set(sum((list(filter(keep, PatchConstructor.closure(dm, e))) for e in exterior_facets), []))
        return exterior_vertices

    @staticmethod
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    def visualise(self, mesh, vertex=0, color="blue"):
        dm = mesh._topology_dm
        (vlo, vhi) = dm.getDepthStratum(0)
        entities = self.callback(dm, vlo+vertex)
        coords = [self.coords(dm, e) for e in entities]
        (x, y) = zip(*coords)
        plt.plot(x, y, "x", markersize=5, color=color)

    @staticmethod
    def get_entities(opts, name, dm):
        sentinel = object()
        codim = opts.getInt("pc_patch_construction_%s_codim" % name, default=sentinel)
        if codim == sentinel:
            dim = opts.getInt("pc_patch_construction_%s_dim" % name, default=0)
            entities = range(*dm.getDepthStratum(dim))
        else:
            entities = range(*dm.getHeightStratum(codim))
        return entities

    def keyfuncs(self, coords):
        sentinel = object()
        sortorders = self.opts.getString("pc_patch_construction_%s_sort_order" % self.name, default=sentinel)
        if sortorders is None or sortorders in [sentinel, "None", ""]:
            return None

        res = []
        for sortorder in sortorders.split("|"):
            sortdata = []
            for axis in sortorder.split(':'):
                ax = int(axis[0])
                if len(axis) > 1:
                    sgn = {'+': 1, '-': -1}[axis[1]]
                else:
                    sgn = 1
                sortdata.append((ax, sgn))

            def keyfunc(z):
                return tuple(sgn*z[1][ax] for (ax, sgn) in sortdata)
            res.append(keyfunc)
        return res

    def __call__(self, pc):
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        opts = PETSc.Options(prefix)
        self.opts = opts
        name = self.name
        assert self.name is not None

        self.set_options(dm, opts, name)

        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = list(filter(select, self.get_entities(opts, name, dm)))

        patches = []
        new_entities = []
        for entity in entities:
            subentities = self.callback(dm, entity)
            if subentities is None:
                continue
            iset = PETSc.IS().createGeneral(subentities, comm=PETSc.COMM_SELF)
            patches.append(iset)
            new_entities.append(entity)

        # Now make the iteration set.

        coords = list(enumerate(self.coords(dm, p) for p in new_entities))



        keyfuncs = self.keyfuncs(coords)

        if keyfuncs is None:
            piterset = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
            return (patches, piterset)

        iterset = []
        for keyfunc in keyfuncs:
            iterset += [x[0] for x in sorted(coords, key=keyfunc)]

        piterset = PETSc.IS().createGeneral(iterset, comm=PETSc.COMM_SELF)
        return (patches, piterset)


class Star(OrderedRelaxation):
    def __init__(self):
        super().__init__()
        self.name = "Star"

    def callback(self, dm, vertex):
        entities = list(self.star(dm, vertex))
        return entities


class MacroStar(OrderedRelaxation):
    def __init__(self):
        super().__init__()
        self.name = "MacroStar"

    def callback(self, dm, vertex):
        if dm.getLabelValue("MacroVertices", vertex) != 1:
            return None
        s = list(self.star(dm, vertex))
        closures = sum((list(self.closure(dm, e)) for e in s), [])

        the_vertices_we_care_about = [v for v in closures if dm.getLabelValue("MacroVertices", v) != 1]
        their_star = sum((list(self.star(dm, v)) for v in the_vertices_we_care_about), [])
        entities = s + their_star
        return entities
