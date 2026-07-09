#!/bin/sh

nvcc -c gpu.cu -o gpu.o
nvcc -O3 gpu.o -o dlink.o -gencode arch=compute_86,code=sm_86 -dlink