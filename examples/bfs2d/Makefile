all: mesh03 mesh06 mesh09 mesh12

mesh12:
	gmsh -2 -format msh2 -clscale 0.12 backwards-facing-step.geo -algo front2d -o coarse12.msh -smooth 3

mesh09:
	gmsh -2 -format msh2 -clscale 0.09 backwards-facing-step.geo -algo front2d -o coarse09.msh -smooth 3

mesh06:
	gmsh -2 -format msh2 -clscale 0.09 backwards-facing-step.geo -algo front2d -o coarse06.msh -smooth 3

mesh03:
	gmsh -2 -format msh2 -clscale 0.09 backwards-facing-step.geo -algo front2d -o coarse03.msh -smooth 3
