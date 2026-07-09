#!/bin/sh

dev=true

nvcc -c gpu.cu -o gpu.o
nvcc -O3 gpu.o -o dlink.o -gencode arch=compute_86,code=sm_86 -dlink

if [[ $dev ]]; then
  ./xdotool key "{ESC}"
  wait
  ./xdotool key "+{F10}"
  wait
fi