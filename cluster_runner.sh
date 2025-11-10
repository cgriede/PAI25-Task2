#!/bin/sh
conda activate PAI2
sbatch -n 8 --mem-per-cpu=2048 --time=4:00:00 --wrap="python -u checker_client.py --results-dir ."