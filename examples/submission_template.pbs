#!/bin/bash --login
#
#        Resource: ARCHER (Cray XC30 (24-core per node))
#    Batch system: PBSPro_select
#
#PBS -N jobalfi
#PBS -A e590
##PBS -q short
#PBS -l walltime=_WALLTIME_
#PBS -l select=_NODES__MEM_
#PBS -m abe
#PBS -M wechsung@maths.ox.ac.uk

# Switch to current working directory
cd $PBS_O_WORKDIR

module unload PrgEnv-cray
module load PrgEnv-gnu
module swap cray-mpich cray-mpich/7.7.4
module swap cray-libsci/16.11.1 cray-libsci/18.12.1
module load cray-hdf5-parallel/1.10.0.1
module unload xalt


FIREDRAKE_DIR=/tmp/firedrake-dev-fmwns

export PETSC_DIR=${FIREDRAKE_DIR}/src/petsc/
export PETSC_ARCH=petsc-gnu51-ivybridge-int32
export CFLAGS="-march=ivybridge -O3"
export PYOP2_CFLAGS="-march=ivybridge -O3"
export PYOP2_SIMD_ISA="avx"
#export PYOP2_LOG_LEVEL="DEBUG"
export OPENBLAS_NUM_THREADS=1
export CC=cc
export CXX=CC
export PYVER=3.6
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GNI_FORK_MODE=FULLCOPY

export PATH=${FIREDRAKE_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${FIREDRAKE_DIR}/lib:/tmp/env368/lib:$LD_LIBRARY_PATH

export PYOP2_CACHE_DIR=$PBS_O_WORKDIR/.cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$PBS_O_WORKDIR/.cache
export XDG_CACHE_HOME=$PBS_O_WORKDIR/.cache
#export PYOP2_CACHE_DIR=/tmp/.cache
#export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/.cache
#export XDG_CACHE_HOME=/tmp/.cache

export TMPDIR=/dev/shm
export TMP=/dev/shm

NODES=_NODES_
PROCS_PER_NODE=_PROCS_PER_NODE_

# Copy to compute nodes and untar venv
aprun -n ${NODES} -N 1 /work/e590/e590/fwe590/copy-firedrake.sh
JOB="_JOB_"
LOG="_LOG_"
aprun -n $((${NODES} * ${PROCS_PER_NODE})) -N ${PROCS_PER_NODE} sh -c "${FIREDRAKE_DIR}/bin/annotate-output ${FIREDRAKE_DIR}/bin/python3 ${PBS_O_WORKDIR}/${JOB}" 2>&1 | tee logs/$LOG.log

