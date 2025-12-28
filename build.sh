#!/bin/bash

rm -rf build

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=OFF \
  -DCUDA_TILE_ENABLE_TESTING=ON \
  -DCUDA_TILE_USE_LLVM_SOURCE_DIR=`pwd`/3rdparty/llvm-project

# cmake --build build 

cmake --build build --target check-cuda-tile

