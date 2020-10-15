exec=mpirun -n 4 python3
pre:
	mkdir -p logs/

iters2dpkp0: pre
	$(exec) iters.py --problem ldc2d --nref-start 1 --nref-end 4 --baseN 16          --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --restriction 2>&1 | tee logs/iters2dldcrestrictnref5.log
	$(exec) iters.py --problem ldc2d --nref-start 1 --nref-end 4 --baseN 16          --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --restriction --singular 2>&1 | tee logs/iters2dldcrestrictsingular.log
	$(exec) iters.py --problem bfs2d --nref-start 1 --nref-end 3 --mesh coarse06.msh --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --smoothing 6 --restriction 2>&1 | tee logs/iters2dbfs06addrestrict6.log
	$(exec) iters.py --problem bfs2d --nref-start 1 --nref-end 3 --mesh coarse03.msh --discretisation pkp0 --mh uniform --k 2 --solver-type almg --stabilisation-type supg --patch star --smoothing 6 --restriction 2>&1 | tee logs/iters2dbfs03addrestrict6.log

iters2dsvlinke:pre
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  planarlattice2d --k 2 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk2planarlattice.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  planarlattice2d --k 3 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk3planarlattice.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  potentialflow2d --k 2 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk2potentialflow.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  potentialflow2d --k 3 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk3potentialflow.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  superposition2d --k 2 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk2superposition.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem  superposition2d --k 3 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvk3superposition.log

iters2dsv: pre
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 2 --baseN 10          --solver-type almg --discretisation sv   --mh bary    --stabilisation-type none --patch macro  --smoothing 6 --restriction --stabilisation-weight 0 2>&1 | tee logs/iters2dsvldcrestrict0k2.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 3 --baseN 10          --solver-type almg --discretisation sv   --mh bary    --stabilisation-type none --patch macro  --smoothing 6 --restriction --stabilisation-weight 0 2>&1 | tee logs/iters2dsvldcrestrict0k3.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 2 --baseN 10          --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvldcrestrict5em3k2.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 3 --baseN 10          --solver-type almg --discretisation sv   --mh bary    --stabilisation-type burman --patch macro  --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvldcrestrict5em3k3.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 2 --baseN 10          --solver-type almg --discretisation th   --mh bary    --stabilisation-type burman --patch macro  --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dthldcrestrict5em3k2.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 3 --baseN 10          --solver-type almg --discretisation th   --mh bary    --stabilisation-type burman --patch macro  --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dthldcrestrict5em3k3.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem ldc2d --k 2 --baseN 16          --solver-type almg --discretisation pkp0 --mh uniform --stabilisation-type supg   --patch star   --smoothing 6 --restriction 2>&1 | tee logs/iters2dsvldccompp2p0restrict.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem bfs2d --k 2 --mesh coarse09.msh --solver-type almg --discretisation sv --mh bary --stabilisation-type none --patch macro --smoothing 6 --restriction --stabilisation-weight 0 2>&1 | tee logs/iters2dsvbfs09restrict0k2.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem bfs2d --k 3 --mesh coarse09.msh --solver-type almg --discretisation sv --mh bary --stabilisation-type none --patch macro --smoothing 6 --restriction --stabilisation-weight 0 2>&1 | tee logs/iters2dsvbfs09restrict0k3.log
	$(exec) iters.py --nref-start 1 --nref-end 4 --problem bfs2d --k 2 --mesh coarse09.msh --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvbfs09restrict5em3k2.log
	$(exec) iters.py --nref-start 1 --nref-end 3 --problem bfs2d --k 3 --mesh coarse09.msh --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --restriction --stabilisation-weight 5e-3 2>&1 | tee logs/iters2dsvbfs09restrict5em3k3.log
	

mms3dpkp0: pre
	$(exec) mms.py --dim 3 --k 1 --nref 5 --discretisation pkp0 --solver-type almg --mh uniform --patch star --gamma 1e4 --stabilisation-type supg --stabilisation--weight 0.05 --baseN 4 2>&1 | tee logs/mms3dpkp0highgamma.log
	$(exec) mms.py --dim 3 --k 1 --nref 5 --discretisation pkp0 --solver-type almg --mh uniform --patch star --gamma 1e0 --stabilisation-type supg --stabilisation--weight 0.05 --baseN 4 2>&1 | tee logs/mms3dpkp0lowgamma.log

mms2dbary: pre
	$(exec) mms.py --dim 2 --nref 6 --k 2 --discretisation sv   --solver-type almg --mh bary --patch macro --gamma 1e4 --stabilisation-type burman --stabilisation-weight 5e-3 2>&1 | tee logs/mms2dbarysv.log
	$(exec) mms.py --dim 2 --nref 6 --k 2 --discretisation pkp0 --solver-type almg --mh bary --patch star  --gamma 1e4 --stabilisation-type burman --stabilisation-weight 5e-3 2>&1 | tee logs/mms2dbarypkp0.log

idealal:
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 0 2>&1 | tee logs/idealalgamma0.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1 2>&1 | tee logs/idealalgamma1.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e1 2>&1 | tee logs/idealalgamma1e1.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e2 2>&1 | tee logs/idealalgamma1e2.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e3 2>&1 | tee logs/idealalgamma1e3.log || 1
	$(exec) iters.py --baseN 16 --nref-start 2 --nref-end 2 --problem ldc2d --k 2 --solver-type allu --discretisation pkp0 --mh uniform --stabilisation-type supg --smoothing 6 --restriction --time --re-max 10000 --gamma 1e4 2>&1 | tee logs/idealalgamma1e4.log || 1
