mesh2dfr:
	gmsh -2 -clscale 0.15 square.geo -algo front2d -o squarefr.msh

mesh2ddel:
	gmsh -2 -clscale 0.15 square.geo -o squaredel.msh

pkp02dstructured:
	python3 graddiv.py --baseN 16 --nref 1 --discretisation pkp0 --dim 2 --diagonal left
	python3 graddiv.py --baseN 16 --nref 2 --discretisation pkp0 --dim 2 --diagonal left
	python3 graddiv.py --baseN 16 --nref 3 --discretisation pkp0 --dim 2 --diagonal left
	#python3 graddiv.py --baseN 16 --nref 4 --discretisation pkp0 --dim 2 --diagonal left

pkp02ddel: mesh2ddel
	python3 graddiv.py --nref 1 --discretisation pkp0 --dim 2 --mesh squaredel.msh
	python3 graddiv.py --nref 2 --discretisation pkp0 --dim 2 --mesh squaredel.msh
	python3 graddiv.py --nref 3 --discretisation pkp0 --dim 2 --mesh squaredel.msh
	#python3 graddiv.py --nref 4 --discretisation pkp0 --dim 2 --mesh squaredel.msh

pkp02dfr: mesh2dfr
	python3 graddiv.py --nref 1 --discretisation pkp0 --dim 2 --mesh squarefr.msh
	python3 graddiv.py --nref 2 --discretisation pkp0 --dim 2 --mesh squarefr.msh
	python3 graddiv.py --nref 3 --discretisation pkp0 --dim 2 --mesh squarefr.msh
	#python3 graddiv.py --nref 4 --discretisation pkp0 --dim 2 --mesh squarefr.msh


mesh3dfr:
	gmsh -3 -clscale 0.8 cube.geo -algo front3d -optimize_netgen -optimize_threshold 0.99 -smooth 3 -o cubefr.msh

mesh3ddel:
	gmsh -3 -clscale 0.8 cube.geo -optimize_netgen -optimize_threshold 0.99 -smooth 3 -o cubedel.msh

pkp03dstructured:
	python3 graddiv.py --nref 2 --k 3 --discretisation pkp0 --dim 3 --baseN 2

pkp03ddel: mesh3ddel
	python3 graddiv.py --nref 1 --k 1 --discretisation pkp0 --dim 3 --mesh cubedel.msh

pkp03dfr: mesh3dfr
	python3 graddiv.py --nref 1 --k 1 --discretisation pkp0 --dim 3 --mesh cubefr.msh
