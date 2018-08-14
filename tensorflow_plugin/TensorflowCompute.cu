// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TensorflowCompute.cuh"


/*! \file TensorflowCompute.cu
    \brief CUDA kernels for TensorflowCompute
*/

// First, the kernel code for zeroing the velocities on the GPU
//! Kernel that zeroes velocities on the GPU
/*! \param d_vel Velocity-mass array from the ParticleData
    \param N Number of particles

    This kernel executes one thread per particle and zeros the velocity of each. It can be run with any 1D block size
    as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__
void gpu_add_scalar4_kernel(Scalar4 *dest, Scalar4 *src, unsigned int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        dest[i].x += src[i].x;
        dest[i].y += src[i].y;
        dest[i].z += src[i].z;
        dest[i].w += src[i].w;
    }
}

cudaError_t gpu_add_scalar4(Scalar4 *dest, Scalar4 *src, unsigned int _N)
{
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_scalar4_kernel<<< grid, threads >>>(dest, src, _N);

    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
}

extern "C" __global__
void gpu_add_virial_kernel(Scalar *dest, Scalar *src, unsigned int _N, unsigned int _pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _N)
    {
        dest[0 * _pitch + i] += src[i * 9 + 0]; //xx
        dest[1 * _pitch + i] += src[i * 9 + 1]; //xy
        dest[2 * _pitch + i] += src[i * 9 + 2]; //xz
        dest[3 * _pitch + i] += src[i * 9 + 4]; //yy
        dest[4 * _pitch + i] += src[i * 9 + 5]; //yz
        dest[5 * _pitch + i] += src[i * 9 + 8]; //zz
    }
}

cudaError_t gpu_add_virial(Scalar *dest, Scalar *src, unsigned int _N, unsigned int _pitch)
{
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_virial_kernel<<< grid, threads >>>(dest, src, _N, _pitch);

    // this method always succeds. If you had a cuda* call in this driver, you could return its error code if not
    // cudaSuccess
    return cudaSuccess;
}