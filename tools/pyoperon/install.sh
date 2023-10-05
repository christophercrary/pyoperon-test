#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX}" ]]; then
    INSTALL_PREFIX=$(python -c "import sysconfig; \
    print(sysconfig.get_config_var('prefix'))")
else
    INSTALL_PREFIX="${CONDA_PREFIX}"
fi

if [[ -z "${INSTALL_PREFIX}" ]]; then
    echo "Error: could not determine install prefix."
    echo $INSTALL_PREFIX
    exit
fi

# Remove `pyoperon` directory, if it exists.
if [ -d pyoperon ]; then
    rm -rf pyoperon
fi

# INSTALL PREREQUISITES...

# export CC=clang
# export CXX=clang++

# aria-csv
git clone https://github.com/AriaFallah/csv-parser csv-parser
mkdir -p ${CONDA_PREFIX}/include/aria-csv
pushd csv-parser
git checkout 4965c9f320d157c15bc1f5a6243de116a4caf101
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf csv-parser

## eve
git clone https://github.com/jfalcou/eve eve
mkdir -p ${CONDA_PREFIX}/include/eve
mkdir -p ${CONDA_PREFIX}/lib
pushd eve
git checkout 3d5821fe770a62c01328b78bb55880b39b8a0a26
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEVE_BUILD_TEST=OFF \
    -DEVE_BUILD_BENCHMARKS=OFF \
    -DEVE_BUILD_DOCUMENTATION=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf eve

## vstat
git clone https://github.com/heal-research/vstat.git
pushd vstat
git switch cpp20-eve
git checkout 5bdb3452cacdb4ac89a82f3c5957307dec088f3c
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf vstat

# Paste installed `vstat` header files out one level, due 
# to some `Operon` testing files using incorrect include paths. 
# (Alternatively, we could overwrite these testing files
# with ones that use correct include paths.)
cp ${CONDA_PREFIX}/include/vstat/vstat/* ${CONDA_PREFIX}/include/vstat/

## fast-float
git clone https://github.com/fastfloat/fast_float.git
pushd fast_float
git checkout 7a6fe5ee799bc5583b9f8ac62966b15d669bed0f
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTLOAT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf fast_float

## pratt-parser
git clone https://github.com/foolnotion/pratt-parser-calculator.git
pushd pratt-parser-calculator
git checkout 025ba103339bb69e3b719b62f3457d5cbb9644e6
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf pratt-parser-calculator

## unordered_dense
git clone https://github.com/martinus/unordered_dense.git
pushd unordered_dense
git checkout 8f10dcc28669abbf45d95fd3af4c00c617af489c
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf unordered_dense

## cpp-sort
git clone https://github.com/Morwenn/cpp-sort.git
pushd cpp-sort
git checkout 29b593a6f9de08281bc5863ca82f6daaf55906d4
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCPPSORT_BUILD_TESTING=OFF \
    -DCPPSORT_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf cpp-sort

## xxhash_cpp
git clone https://github.com/RedSpah/xxhash_cpp.git
pushd xxhash_cpp
git checkout 2400ea5adc1156b586ee988ea7850be45d9011b5
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
cmake --install build
popd
rm -rf xxhash_cpp

## fmt (conda only includes the shared library and we want the static)
git clone https://github.com/fmtlib/fmt.git
pushd fmt
git checkout a33701196adfad74917046096bf5a2aa0ab0bb50
mkdir build
pushd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFMT_TEST=OFF \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
make install
popd
popd
rm -rf fmt

# # taskflow
# git clone https://github.com/taskflow/taskflow.git
# pushd taskflow
# git checkout 12f8bd4e970ab27fd3dee3bffa24b5b48b54ba39
# mkdir build
# cmake -S . -B build \
#     -DTF_BUILD_EXAMPLES=OFF \
#     -DTF_BUILD_TESTS=OFF \
#     -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
# cmake --install build
# popd
# rm -rf taskflow

# operon
git clone https://github.com/heal-research/operon.git
pushd operon
git switch cpp20
git checkout 303060d38d070b7fb322cc8e88b2ec44f2925657
mkdir build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SINGLE_PRECISION=ON \
    -DUSE_CERES=ON \
    -DBUILD_CLI_PROGRAMS=OFF \
    -Doperon_DEVELOPER_MODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_FLAGS="-march=native -fno-math-errno" \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/lib64/cmake
cmake --build build -j -t operon_operon
cmake --install build

popd
rm -rf operon

# Build `pyoperon`.
git clone https://github.com/heal-research/pyoperon.git
pushd pyoperon
git switch cpp20
git checkout a37d3b6edc88989ea29d12e0b4052000494c62cf
pip install .
popd
rm -rf pyoperon