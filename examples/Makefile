iters:
	mpiexec -n 4 python3 iters.py --baseN 16 --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --problem ldc2d

mms2dpkp0:
	#mpiexec -n 4 python3 mms.py --dim 2 --nref 2 --discretisation pkp0 --solver-type allu --mh uniform --patch star --gamma 1e4 --stabilisation-type supg --baseN 10
	mpiexec -n 4 python3 mms.py --nref 2 --k 2 --discretisation pkp0 --mh uniform --patch star --solver-type allu --clear --dim 2 --baseN 10

mms2dsv:
	mpiexec -n 4 python3 mms.py --dim 2 --nref 4 --k 2 --discretisation sv   --solver-type almg --mh bary --patch macro --gamma 1e4 --stabilisation-type burman
	mpiexec -n 4 python3 mms.py --dim 2 --nref 4 --k 2 --discretisation pkp0 --solver-type almg --mh bary --patch star  --gamma 1e4 --stabilisation-type burman
