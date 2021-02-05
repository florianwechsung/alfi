# (A)ugmented (L)agrangian based solvers for the Navier Stokes equation in (Fi)redrake

This repository contains implementations for scalable solvers for the steady Navier-Stokes equations, implemented in Firedrake. These solvers combine an augmented Lagrangian approach for the Schur complement with a custom robust multigrid scheme for the top-left block of the mixed system.

Currently we implement the following discretisations

- `[P_2]^2-P_0`, `[P_1+FacetBubble]^3-P_0`, `[P_2+FacetBubble]^3-P_0` on simplices as described in [Farrell, Mitchell, Wechsung, _An Augmented Lagrangian Preconditioner for the 3D Stationary Incompressible Navier--Stokes Equations at High Reynolds Number_](https://epubs.siam.org/doi/abs/10.1137/18M1219370)
- `[P_k]^d-P_{k-1}^{DG}` Scott Vogelius elements and `[P_k]^d-P_{k-1}` Taylor-Hood elements for on barycentrically refined simplices for `k⩾2` as described in [Farrell, Mitchell, Scott, Wechsung, _A Reynolds-robust preconditioner for the Scott-Vogelius discretization of the stationary incompressible Navier-Stokes equations_](https://arxiv.org/abs/2004.09398)

Solvers for `H(div)-L2` discretisations on both simplices and quads/hexes as well as solvers for `Q_k-Q_{k-2}^{DG}` and `Q_k-P_{k-1}^{DG}` discretisations on quads/hexes exist on branches and will be available soon.

# Developers

Main development by 

- [Florian Wechsung](https://florianwechsung.github.io)
- [Patrick Farrell](https://pefarrell.org)  
- [Lawrence Mitchell](https://www.dur.ac.uk/research/directory/staff/?mode=staff&id=17243)

and with contributions by

- [Hamza Alawiye](https://www.maths.ox.ac.uk/people/hamza.alawiye)
