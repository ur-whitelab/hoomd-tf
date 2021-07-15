// Copyright (c) 2020 HOOMD-TF Developers

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

extern "C" __global__
void htf_gpu_copy3_kerenl(Scalar4 *dest_array, Scalar4* src_array, unsigned int N)
    {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N)
    {
        dest_array[i].x = src_array.data[i].x;
        dest_array[i].y = src_array.data[i].y;
        dest_array[i].z = src_array.data[i].z;
    }

    }

cudaError_t htf_gpu_copy3(Scalar4 *dest_array, Scalar4* src_array, , unsigned int m_N, cudaStream_t s)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)m_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    htf_gpu_copy3_kerenl<<< grid, threads, 0, s >>>(dest_array, src_array, m_N);

    // this method always succeds.
    return cudaSuccess;
    }