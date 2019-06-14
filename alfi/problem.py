from firedrake import *

class NavierStokesProblem(object):

    # This overlap is a bit too massive!
    distribution_parameters = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

    def mesh(self, distribution_parameters):
        raise NotImplementedError

    def bcs(self, Z):
        raise NotImplementedError

    def has_nullspace(self):
        raise NotImplementedError

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        else:
            return None

    def char_velocity(self):
        return 1.0

    def char_length(self):
        return 1.0

    def mesh_size(self, u):
        return CellSize(u.ufl_domain())

    def rhs(self, Z):
        return None

    def relaxation_direction(self):
        return None
