#!/bin/bash

WORKDIR=`git rev-parse --show-toplevel`

${WORKDIR}/build/bin/cuda-tile-opt example.mlir --mlir-print-ir-after-all -cse

${WORKDIR}/build/bin/cuda-tile-translate example.mlir --bytecode-version=13.1 --mlir-to-cudatilebc --no-implicit-module -o example.tilebc
g++ example.cpp -o example -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
