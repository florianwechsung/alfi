all: mesh13 mesh45 mesh60 

mesh13:
	gmsh -3 -format msh2 -clscale 0.13 backwards-facing-step-3d.geo -algo front3d -o coarse13.msh -optimize_netgen -smooth 3

mesh45:
	gmsh -3 -format msh2 -clscale 0.45 backwards-facing-step-3d.geo -algo front3d -o coarse45.msh -optimize_netgen -smooth 3

mesh60:
	gmsh -3 -format msh2 -clscale 0.60 backwards-facing-step-3d.geo -algo front3d -o coarse60.msh -optimize_netgen -smooth 3
