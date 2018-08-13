// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _TENSORFLOW_COMPUTE_CUH_
#define _TENSORFLOW_COMPUTE_CUH_

#include "TensorflowCompute.h"

extern "C" cudaError_t gpu_add_scalar4(Scalar4 *dest, Scalar4 *src, unsigned int N);
extern "C" cudaError_t gpu_add_virial(Scalar4 *dest, Scalar4 *src, unsigned int N, unsigned int pitch);

template<>
void receiveForcesFunctorAdd::operator()<IPCCommMode::GPU>(Scalar4* dest, Scalar4* src)
{
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_scalar4<<< grid, threads >>>(dest, src, _N);

    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
}

template<>
void receiveVirialFunctorAddCPU::operator()<IPCCommMode::GPU>(Scalar4* dest, Scalar4* src)
{
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_virial<<< grid, threads >>>(dest, src, _N, _pitch);

    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
}

#endif // _TENSORFLOW_COMPUTE_CUH_
