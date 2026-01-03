#!/bin/bash

WORKDIR=`git rev-parse --show-toplevel`

${WORKDIR}/build/bin/cuda-tile-translate example.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o example.tilebc
tileiras --gpu-name sm_120 example.tilebc -o example.tilebc # (Optional: on the physical machine) please run this if you run on the gpu runtime container
g++ example.cpp -o example -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
