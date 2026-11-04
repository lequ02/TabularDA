#!/usr/bin/env bash
set -euo pipefail
cd /home/thuy/Research/minh_data_synth/TabularDA/output/comparison_devf1_20261015_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec /home/thuy/miniconda3/envs/env/bin/python -u parallel_queue.py
