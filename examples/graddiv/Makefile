exec=mpiexec -n 6 python3

pkp02dcomparison:
	mkdir -p logs
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 2 --dim 2 --smoother amg 2>&1 | tee    logs/pkp02dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 2 --dim 2 --smoother amg 2>&1 | tee -a logs/pkp02dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 2 --dim 2 --smoother amg 2>&1 | tee -a logs/pkp02dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation pkp0 --k 2 --dim 2 --smoother amg 2>&1 | tee -a logs/pkp02dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 2 --dim 2 --smoother patch --transfer 2>&1 | tee    logs/pkp02dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 2 --dim 2 --smoother patch --transfer 2>&1 | tee -a logs/pkp02dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 2 --dim 2 --smoother patch --transfer 2>&1 | tee -a logs/pkp02dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation pkp0 --k 2 --dim 2 --smoother patch --transfer 2>&1 | tee -a logs/pkp02dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 2 --dim 2 --smoother patch 2>&1 | tee    logs/pkp02dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 2 --dim 2 --smoother patch 2>&1 | tee -a logs/pkp02dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 2 --dim 2 --smoother patch 2>&1 | tee -a logs/pkp02dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation pkp0 --k 2 --dim 2 --smoother patch 2>&1 | tee -a logs/pkp02dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi --transfer 2>&1 | tee    logs/pkp02dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi --transfer 2>&1 | tee -a logs/pkp02dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi --transfer 2>&1 | tee -a logs/pkp02dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi --transfer 2>&1 | tee -a logs/pkp02dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi 2>&1 | tee    logs/pkp02dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi 2>&1 | tee -a logs/pkp02dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi 2>&1 | tee -a logs/pkp02dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation pkp0 --k 2 --dim 2 --smoother jacobi 2>&1 | tee -a logs/pkp02dcomparison-jacobi-notransfer.log

pkp03dcomparison:
	mkdir -p logs
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 1 --dim 3 --smoother amg 2>&1 | tee    logs/pkp03dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 1 --dim 3 --smoother amg 2>&1 | tee -a logs/pkp03dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 1 --dim 3 --smoother amg 2>&1 | tee -a logs/pkp03dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 1 --dim 3 --smoother patch --transfer 2>&1 | tee    logs/pkp03dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 1 --dim 3 --smoother patch --transfer 2>&1 | tee -a logs/pkp03dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 1 --dim 3 --smoother patch --transfer 2>&1 | tee -a logs/pkp03dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 1 --dim 3 --smoother patch 2>&1 | tee    logs/pkp03dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 1 --dim 3 --smoother patch 2>&1 | tee -a logs/pkp03dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 1 --dim 3 --smoother patch 2>&1 | tee -a logs/pkp03dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi --transfer 2>&1 | tee    logs/pkp03dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi --transfer 2>&1 | tee -a logs/pkp03dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi --transfer 2>&1 | tee -a logs/pkp03dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi 2>&1 | tee    logs/pkp03dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi 2>&1 | tee -a logs/pkp03dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation pkp0 --k 1 --dim 3 --smoother jacobi 2>&1 | tee -a logs/pkp03dcomparison-jacobi-notransfer.log

sv2dcomparison:
	mkdir -p logs
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother amg 2>&1 | tee    logs/sv2dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother amg 2>&1 | tee -a logs/sv2dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother amg 2>&1 | tee -a logs/sv2dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother amg 2>&1 | tee -a logs/sv2dcomparison-amg.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee    logs/sv2dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee -a logs/sv2dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee -a logs/sv2dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee -a logs/sv2dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch 2>&1 | tee    logs/sv2dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch 2>&1 | tee -a logs/sv2dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch 2>&1 | tee -a logs/sv2dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother patch 2>&1 | tee -a logs/sv2dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee    logs/sv2dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee -a logs/sv2dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee -a logs/sv2dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee -a logs/sv2dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 4 --nref 1 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi 2>&1 | tee    logs/sv2dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 2 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi 2>&1 | tee -a logs/sv2dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 3 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi 2>&1 | tee -a logs/sv2dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 4 --nref 4 --discretisation sv --k 2 --dim 2 --mh bary --patch macro --smoother jacobi 2>&1 | tee -a logs/sv2dcomparison-jacobi-notransfer.log

sv3dcomparison:
	mkdir -p logs
	$(exec) graddiv.py --baseN 2 --nref 1 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother amg 2>&1 | tee    logs/sv3dcomparison-amg.log
	$(exec) graddiv.py --baseN 2 --nref 2 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother amg 2>&1 | tee -a logs/sv3dcomparison-amg.log
	$(exec) graddiv.py --baseN 2 --nref 3 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother amg 2>&1 | tee -a logs/sv3dcomparison-amg.log
	$(exec) graddiv.py --baseN 2 --nref 1 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee    logs/sv3dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 2 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee -a logs/sv3dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 3 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch --transfer 2>&1 | tee -a logs/sv3dcomparison-patch-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 1 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch 2>&1 | tee    logs/sv3dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 2 --nref 2 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch 2>&1 | tee -a logs/sv3dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 2 --nref 3 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother patch 2>&1 | tee -a logs/sv3dcomparison-patch-notransfer.log
	$(exec) graddiv.py --baseN 2 --nref 1 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee    logs/sv3dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 2 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee -a logs/sv3dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 3 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi --transfer 2>&1 | tee -a logs/sv3dcomparison-jacobi-transfer.log
	$(exec) graddiv.py --baseN 2 --nref 1 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi 2>&1 | tee    logs/sv3dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 2 --nref 2 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi 2>&1 | tee -a logs/sv3dcomparison-jacobi-notransfer.log
	$(exec) graddiv.py --baseN 2 --nref 3 --discretisation sv --k 3 --dim 3 --mh bary --patch macro --smoother jacobi 2>&1 | tee -a logs/sv3dcomparison-jacobi-notransfer.log
