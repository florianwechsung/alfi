exec=mpirun -n 12 python3
pre:
	mkdir -p logs/

iters2dpkp0: pre
	#$(exec) iters.py --problem ldc2d --nref 2 --baseN 16         --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star
	$(exec) iters.py --problem bfs2d --nref 3 --mesh coarse06.msh --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star 2>&1 | tee logs/iters2dbfs06.log
	$(exec) iters.py --problem bfs2d --nref 3 --mesh coarse06.msh --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --patch-composition multiplicative 2>&1 | tee logs/iters2dbfs06mult.log

iters2dsv: pre
	#$(exec) iters.py --nref 2 --problem ldc2d --k 2 --baseN 10         --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  2>&1 | tee logs/iters2dsvldc.log
	#$(exec) iters.py --nref 2 --problem ldc2d --k 2 --baseN 16         --solver-type almg --discretisation pkp0 --mh uniform --stabilisation-type supg   --patch star   2>&1 | tee logs/iters2dsvldccompp2p0.log
	$(exec) iters.py --nref 2 --problem bfs2d --k 2 --mesh coarse12.msh --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  2>&1 | tee logs/iters2dsvbfs12.log
	$(exec) iters.py --nref 2 --problem bfs2d --k 2 --mesh coarse09.msh --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  2>&1 | tee logs/iters2dsvbfs09.log
	$(exec) iters.py --nref 2 --problem bfs2d --k 2 --mesh coarse06.msh --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  2>&1 | tee logs/iters2dsvbfs06.log

mms2dpkp0: pre
	#$(exec) mms.py --dim 2 --nref 2 --discretisation pkp0 --solver-type allu --mh uniform --patch star --gamma 1e4 --stabilisation-type supg --baseN 10
	$(exec) mms.py --nref 2 --k 2 --discretisation pkp0 --mh uniform --patch star --solver-type allu --clear --dim 2 --baseN 10

mms2dbary: pre
	$(exec) mms.py --dim 2 --nref 4 --k 2 --discretisation sv   --solver-type almg --mh bary --patch macro --gamma 1e4 --stabilisation-type burman 2>&1 | tee logs/mms2dbarysv.log
	$(exec) mms.py --dim 2 --nref 4 --k 2 --discretisation pkp0 --solver-type almg --mh bary --patch star  --gamma 1e4 --stabilisation-type burman 2>&1 | tee logs/mms2dbarypkp0.log
