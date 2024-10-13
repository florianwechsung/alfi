from firedrake import *
from functools import partial

""" For the P1 + FacetBubble space in 3D, the standard prolongation does not
preserve the flux across coarse grid facets.  We fix this with a manual
scaling. """


class BubbleTransfer(object):
    def __new__(cls, Vc, Vf):
        if cls is not BubbleTransfer:
            return super().__new__(cls)

        fe = Vc.finat_element
        entity_dofs = fe.entity_dofs()
        dim = fe.cell.get_spatial_dimension()
        if fe.value_shape == (dim,) and len(entity_dofs[dim-1][0]) == 1:
            return NormalBubbleTransfer(Vc, Vf)

        return super().__new__(cls)

    def __init__(self, Vc, Vf):
        meshc = Vc.mesh()
        meshf = Vf.mesh()
        self.Vc = Vc
        self.Vf = Vf
        P1c = VectorFunctionSpace(meshc, "CG", 1)
        FBc = VectorFunctionSpace(meshc, "FacetBubble", 3)
        P1f = VectorFunctionSpace(meshf, "CG", 1)
        FBf = VectorFunctionSpace(meshf, "FacetBubble", 3)

        self.p1c = Function(P1c)
        self.fbc = Function(FBc)
        self.p1f = Function(P1f)
        self.fbf = Function(FBf)
        self.rhs = Function(FBc)

        trial = TrialFunction(FBc)
        test = TestFunction(FBc)
        n = FacetNormal(meshc)
        a = inner(inner(trial, n), inner(test, n)) + inner(trial - inner(trial, n)*n, test-inner(test, n)*n)
        a = a*ds + avg(a)*dS
        A = assemble(a).M.handle.getDiagonal()
        # A is diagonal, so "solve" by dividing by the diagonal.
        ainv = A.copy()
        ainv.reciprocal()
        self.ainv = ainv
        L = inner(inner(self.fbc, n)/0.625, inner(test, n)) + inner(self.fbc - inner(self.fbc, n)*n, test-inner(test, n)*n)
        L = L*ds + avg(L) * dS
        self.assemble_rhs = partial(assemble, L, tensor=self.rhs)

        # TODO: Parameterise these by the element definition.
        # These are correct for P1 + FB in 3D.
        # Dof layout for the combined element on the reference cell is:
        # 4 vertex dofs, then 4 face dofs.
        """
        These two kernels perform a change of basis from a nodal to a
        hierachichal basis ("split") and then back from a hierachichal
        basis to a nodal basis ("combine").  The nodal functionspace is
        given by P1FB, the hierachichal one is given by two spaces: P1 and
        FB. The splitting kernel is defined so that
            [a b] * [p1fb_1; ...;p1fb_8] = [p1_1;...;p1_4;fb_1;...;fb_4]
        and the combine kernel is defined so that
            [a;b] * [p1_1;...;p1_4;fb_1;...;fb_4] = [p1fb_1;...;p1fb_8]

        Notation: [x;y]: vertically stack x and y, [x y]: horizontally stacked
        """
        self.split_kernel = op2.Kernel("""
void split(double p1[12], double fb[12], const double both[24]) {
//  for (int i = 0; i < 12; i++) {
//    p1[i] = 0;
//    fb[i] = 0;
//  }

  double a[8][4] = {{1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.}};

  double b[8][4] = {{0., -1./3, -1./3, -1./3},
                    {-1./3, 0., -1./3, -1./3},
                    {-1./3, -1./3, 0., -1./3},
                    {-1./3, -1./3, -1./3, 0.},
                    {1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.}};


  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        p1[3 * i + j] += a[k][i] * both[3 * k + j];
        fb[3 * i + j] += b[k][i] * both[3 * k + j];
      }
    }
  }
}""", "split")

        self.split_kernel_adj = op2.Kernel("""
void splitadj(const double p1[12], const double fb[12], double both[24]) {
  //for (int i = 0; i < 24; i++) both[i] = 0;

  double a[8][4] = {{1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.},
                    {0., 0., 0., 0.}};

  double b[8][4] = {{0., -1./3, -1./3, -1./3},
                    {-1./3, 0., -1./3, -1./3},
                    {-1./3, -1./3, 0., -1./3},
                    {-1./3, -1./3, -1./3, 0.},
                    {1., 0., 0., 0.},
                    {0., 1., 0., 0.},
                    {0., 0., 1., 0.},
                    {0., 0., 0., 1.}};


  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 3; j++) {
        both[3 * k + j] += a[k][i] * p1[3*i+j];
        both[3 * k + j] += b[k][i] * fb[3*i+j];
      }
    }
  }
}""", "splitadj")
        self.combine_kernel = op2.Kernel("""
void combine(const double p1[12], const double fb[12], double both[24]) {
  //for (int i = 0; i < 24; i++) both[i] = 0;

  double a[4][8] = {{1., 0., 0., 0., 0.,    1./3., 1./3., 1./3.},
                    {0., 1., 0., 0., 1./3., 0.,    1./3., 1./3.},
                    {0., 0., 1., 0., 1./3., 1./3., 0.,    1./3.},
                    {0., 0., 0., 1., 1./3., 1./3., 1./3., 0.   }};
  double b[4][8] = {{0., 0., 0., 0., 1., 0., 0., 0.},
                    {0., 0., 0., 0., 0., 1., 0., 0.},
                    {0., 0., 0., 0., 0., 0., 1., 0.},
                    {0., 0., 0., 0., 0., 0., 0., 1.}};

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 3; j++) {
        both[3 * i + j] += a[k][i] * p1[3 * k + j];
        both[3 * i + j] += b[k][i] * fb[3 * k + j];
      }
    }
  }
}
""", "combine")

        self.combine_kernel_adj = op2.Kernel("""
void combineadj(const double both[24], double p1[12], double fb[12]) {
  //for (int i = 0; i < 12; i++) {
  //  p1[i] = 0;
  //  fb[i] = 0;
  //}

  double a[4][8] = {{1., 0., 0., 0., 0.,    1./3., 1./3., 1./3.},
                    {0., 1., 0., 0., 1./3., 0.,    1./3., 1./3.},
                    {0., 0., 1., 0., 1./3., 1./3., 0.,    1./3.},
                    {0., 0., 0., 1., 1./3., 1./3., 1./3., 0.   }};
  double b[4][8] = {{0., 0., 0., 0., 1., 0., 0., 0.},
                    {0., 0., 0., 0., 0., 1., 0., 0.},
                    {0., 0., 0., 0., 0., 0., 1., 0.},
                    {0., 0., 0., 0., 0., 0., 0., 1.}};

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 3; j++) {
        p1[3 * k + j] += a[k][i] * both[3 * i + j];
        fb[3 * k + j] += b[k][i] * both[3 * i + j];
      }
    }
  }
}
""", "combineadj")
        self.count_kernel = op2.Kernel("""
void count(double both[24], double fb[12], double p1[12]) {
  for (int i = 0; i < 24; i++) {
    both[i] += 1;
  }
  for (int i = 0; i < 12; i++) {
    fb[i] += 1;
    p1[i] += 1;
  }
}
""", "count")
        self.countp1f = Function(self.p1f.function_space())
        self.countfbf = Function(self.fbf.function_space())
        self.countvf = Function(self.Vf)
        self.countp1c = Function(self.p1c.function_space())
        self.countfbc = Function(self.fbc.function_space())
        self.countvc = Function(self.Vc)
        op2.par_loop(self.count_kernel, self.countp1f.ufl_domain().cell_set,
                     self.countvf.dat(op2.INC, self.countvf.cell_node_map()),
                     self.countfbf.dat(op2.INC, self.countfbf.cell_node_map()),
                     self.countp1f.dat(op2.INC, self.countp1f.cell_node_map())
                     )
        op2.par_loop(self.count_kernel, self.countp1c.ufl_domain().cell_set,
                     self.countvc.dat(op2.INC, self.countvc.cell_node_map()),
                     self.countfbc.dat(op2.INC, self.countfbc.cell_node_map()),
                     self.countp1c.dat(op2.INC, self.countp1c.cell_node_map())
                     )


    def restrict(self, fine, coarse):
        # All the adjoint steps of prolong, but in reverse
        with self.p1f.dat.vec_wo as v:
            v.zeroEntries()
        with self.fbf.dat.vec_wo as v:
            v.zeroEntries()
        fine.dat.data[:, :] /= self.countvf.dat.data[:, :]
        op2.par_loop(self.combine_kernel_adj, fine.ufl_domain().cell_set,
                     fine.dat(op2.READ, fine.cell_node_map()),
                     self.p1f.dat(op2.INC, self.p1f.cell_node_map()),
                     self.fbf.dat(op2.INC, self.fbf.cell_node_map())
                     )

        restrict(self.p1f, self.p1c)
        restrict(self.fbf, self.fbc)

        self.assemble_rhs()
        with self.fbc.dat.vec_wo as fbc, self.rhs.dat.vec_ro as b:
            fbc.pointwiseMult(b, self.ainv)

        with coarse.dat.vec_wo as v:
            v.zeroEntries()
        self.p1c.dat.data[:, :] /= self.countp1c.dat.data[:, :]
        self.fbc.dat.data[:, :] /= self.countfbc.dat.data[:, :]
        op2.par_loop(self.split_kernel_adj, coarse.ufl_domain().cell_set,
                     self.p1c.dat(op2.READ, self.p1c.cell_node_map()),
                     self.fbc.dat(op2.READ, self.fbc.cell_node_map()),
                     coarse.dat(op2.INC, coarse.cell_node_map()))

    def prolong(self, coarse, fine):
        # Step 1: perform a change of basis into a piecewise linear and into a bubble function
        with self.p1c.dat.vec_wo as v:
            v.zeroEntries()
        with self.fbc.dat.vec_wo as v:
            v.zeroEntries()
        op2.par_loop(self.split_kernel, coarse.ufl_domain().cell_set,
                     self.p1c.dat(op2.INC, self.p1c.cell_node_map()),
                     self.fbc.dat(op2.INC, self.fbc.cell_node_map()),
                     coarse.dat(op2.READ, coarse.cell_node_map()))
        self.p1c.dat.data[:, :] /= self.countp1c.dat.data[:, :]
        self.fbc.dat.data[:, :] /= self.countfbc.dat.data[:, :]

        # Step 2: Piecewise linear functions are prolonged correctly, but for bubble functions
        # we make a mistake. This is because the bubble function can't be represented perfectly.
        # It turns out the mistake is exactly that we underestimate the flux across the coarse facets
        # by a factor of 0.625. To fix this, we scale up the normal components of the bubble functions,
        # while keeping the tangential components unchanged.
        self.assemble_rhs()
        with self.fbc.dat.vec_wo as fbc, self.rhs.dat.vec_ro as b:
            fbc.pointwiseMult(b, self.ainv)

        # Step 3: Now prolong the two functions and then combine back together again.
        prolong(self.p1c, self.p1f)
        prolong(self.fbc, self.fbf)

        with fine.dat.vec_wo as v:
            v.zeroEntries()
        op2.par_loop(self.combine_kernel, fine.ufl_domain().cell_set,
                     self.p1f.dat(op2.READ, self.p1f.cell_node_map()),
                     self.fbf.dat(op2.READ, self.fbf.cell_node_map()),
                     fine.dat(op2.INC, fine.cell_node_map()))
        fine.dat.data[:, :] /= self.countvf.dat.data[:, :]


class NormalBubbleTransfer(BubbleTransfer):
    manager = TransferManager()

    def __init__(self, Vc, Vf):
        W = Vc
        depth = W.mesh().topological_dimension() - 1

        mesh_dm = W.mesh().topology_dm
        W_local_ises_indices = W.dof_dset.local_ises[0].indices
        section = W.dm.getDefaultSection()
        indices = []
        for p in range(*mesh_dm.getDepthStratum(depth)):
            dof = section.getDof(p)
            if dof <= 0:
                continue
            off = section.getOffset(p)
            W_indices = slice(off*W.value_size, W.value_size * (off + dof))
            indices.extend(W_local_ises_indices[W_indices])

        ainv = W.dof_dset.layout_vec.copy()
        ainv.set(1)
        ainv[indices] = 1 / 0.625
        self.ainv = ainv

        self.primal = Function(W)
        self.dual = Cofunction(W.dual())

    def restrict(self, rf, rc):
        self.manager.restrict(rf, self.dual)
        with self.dual.dat.vec_wo as wc, rc.dat.vec_ro as b:
            wc.pointwiseMult(b, self.ainv)

    def prolong(self, uc, uf):
        with self.primal.dat.vec_wo as wc, uc.dat.vec_ro as b:
            wc.pointwiseMult(b, self.ainv)
        self.manager.prolong(self.primal, uf)
