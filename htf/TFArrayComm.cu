// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#include "TFArrayComm.cuh"

/*! \file TFArrayComm.cu
    \brief CUDA kernels and functions for TFArrayComm
*/

extern "C" __global__
void htf_gpu_unstuff4_kerenl(Scalar4 *array, unsigned int N)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N)
        array[i].w = static_cast<Scalar> (__scalar_as_int(array[i].w));
    }

cudaError_t htf_gpu_unstuff4(Scalar4 *array, unsigned int m_N, cudaStream_t s)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)m_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    htf_gpu_unstuff4_kerenl<<< grid, threads, 0, s >>>(array, m_N);

    // this method always succeds.
    return cudaSuccess;
    }