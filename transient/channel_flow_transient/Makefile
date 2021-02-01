all: mesh03 mesh04 mesh06 mesh09 mesh12

mesh12:
        gmsh -2 -format msh2 -clscale 0.12 channel.geo -algo front2d -o coarse12.msh -smooth 3

mesh09:
        gmsh -2 -format msh2 -clscale 0.09 channel.geo -algo front2d -o coarse09.msh -smooth 3

mesh08:
        gmsh -2 -format msh2 -clscale 0.08 channel.geo -algo front2d -o coarse08.msh -smooth 3

mesh06:
        gmsh -2 -format msh2 -clscale 0.06 channel.geo -algo front2d -o coarse06.msh -smooth 3

mesh04:
        gmsh -2 -format msh2 -clscale 0.04 channel.geo -algo front2d -o coarse04.msh -smooth 3

mesh03:
        gmsh -2 -format msh2 -clscale 0.03 channel.geo -algo front2d -o coarse03.msh -smooth 3

