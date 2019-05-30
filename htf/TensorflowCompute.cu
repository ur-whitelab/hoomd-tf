// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#include "TensorflowCompute.cuh"
#include <iostream>


/*! \file TensorflowCompute.cu
    \brief CUDA kernels and functions for TensorflowCompute
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

cudaError_t gpu_add_scalar4(Scalar4 *dest, Scalar4 *src, unsigned int _N, cudaStream_t s)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_scalar4_kernel<<< grid, threads, 0, s >>>(dest, src, _N);

    // this method always succeds.
    // If you had a cuda* call in this driver, you could return its error code, if not
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

cudaError_t gpu_add_virial(Scalar *dest, Scalar *src, unsigned int _N, unsigned int _pitch, cudaStream_t s)
    {
    // setup the grid to run the kernel
    int block_size = 256;
    dim3 grid( (int)ceil((double)_N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);

    // run the kernel
    gpu_add_virial_kernel<<< grid, threads, 0, s >>>(dest, src, _N, _pitch);

    // this method always succeds.
    // If you had a cuda* call in this driver, you could return its error code, if not
    // cudaSuccess
    return cudaSuccess;
    }

#include "hoomd/TextureTools.h"
#include "hoomd/Index1D.h"
#include <assert.h>

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;

//! Texture for reading the neighbor list
texture<unsigned int, 1, cudaReadModeElementType> nlist_tex;

template<unsigned char use_gmem_nlist>
__global__ void gpu_reshape_nlist_kernel(Scalar4* dest,
                                         const unsigned int N,
                                         const unsigned int NN,
                                         const unsigned int offset,
                                         const unsigned int batch_size,
                                         const Scalar4 *d_pos,
                                         const BoxDim box,
                                         const unsigned int *d_n_neigh,
                                         const unsigned int *d_nlist,
                                         const unsigned int *d_head_list,
                                         double rmax)
    {

    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx >= N || idx - offset >= batch_size)
        return;

    // load in the length of the list
    unsigned int n_neigh = d_n_neigh[idx];
    const unsigned int head_idx = d_head_list[idx];

    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int typei = __scalar_as_int(postype.w);

    // prefetch neighbor index
    unsigned int cur_neigh = 0;
    unsigned int next_neigh(0);
    if (use_gmem_nlist)
        next_neigh = d_nlist[head_idx];
    else
        next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx);

    // loop over neighbors
    assert(n_neigh <= NN);
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
        {

        // read the current neighbor index
        // prefetch the next value and set the current one
        cur_neigh = next_neigh;
        if (use_gmem_nlist)
            next_neigh = d_nlist[head_idx + neigh_idx + 1];
        else
            next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idx+1);

        // get the neighbor's position
        Scalar4 neigh_postype = texFetchScalar4(d_pos, pdata_pos_tex, cur_neigh);
        Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

        // calculate dr (with periodic boundary conditions)
        Scalar3 dx = neigh_pos - pos;

        // apply periodic boundary conditions
        dx = box.minImage(dx);

        // access needed parameters
        unsigned int typej = __scalar_as_int(neigh_postype.w);

        // calculate r
        Scalar rsq = dot(dx, dx);

        if (rsq < (rmax * rmax))
        {
            dest[(idx - offset) * NN + neigh_idx].x = dx.x;
            dest[(idx - offset) * NN + neigh_idx].y = dx.y;
            dest[(idx - offset) * NN + neigh_idx].z = dx.z;
            dest[(idx - offset) * NN + neigh_idx].w = static_cast<Scalar> (typej);

            }
        }
    }


cudaError_t gpu_reshape_nlist(Scalar4* dest,
			      const Scalar4 *d_pos,
			      const unsigned int N,
                  	      const unsigned int NN,
                  	      const unsigned int offset,
                  	      const unsigned int batch_size,
			      const unsigned int n_ghost,
			      const BoxDim& box,
			      const unsigned int *d_n_neigh,
			      const unsigned int *d_nlist,
			      const unsigned int *d_head_list,
			      const unsigned int size_nlist,
			      const unsigned int block_size,
			      const unsigned int compute_capability,
			      const unsigned int max_tex1d_width,
			      double rmax,
			      cudaStream_t stream)
    {

    assert(d_pos);
    assert(dest);
    assert(d_n_neigh);
    assert(d_nlist);
    assert(d_head_list);

    //set neighbors to zeros
    cudaMemset(dest, 1, batch_size * NN * sizeof(Scalar4));

    // texture bind
    if (compute_capability < 350)
        {
        // bind the pdata position texture
        pdata_pos_tex.normalized = false;
        pdata_pos_tex.filterMode = cudaFilterModePoint;
        cudaError_t error = cudaBindTexture(0,
                                            pdata_pos_tex,
                                            d_pos,
                                            sizeof(Scalar4) * (N+n_ghost));
        if (error != cudaSuccess)
            return error;

        if (size_nlist <= max_tex1d_width)
            {
            nlist_tex.normalized = false;
            nlist_tex.filterMode = cudaFilterModePoint;
            error = cudaBindTexture(0, nlist_tex, d_nlist, sizeof(unsigned int)*size_nlist);
            if (error != cudaSuccess)
                return error;
            }
        }

    if (compute_capability < 350 && size_nlist > max_tex1d_width)
        {
        // use global memory when the neighbor list must be texture bound,
        // but exceeds the max size of a texture
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_reshape_nlist_kernel<1>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( batch_size / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);

        gpu_reshape_nlist_kernel<1><<< grid, threads, 0, stream>>>(dest,
            N,
            NN,
            offset,
            batch_size,
            d_pos,
            box,
            d_n_neigh,
            d_nlist,
            d_head_list,
            rmax);
    }
    else
    {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_reshape_nlist_kernel<0>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int run_block_size = min(block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( batch_size / run_block_size + 1, 1, 1);
        dim3 threads(run_block_size, 1, 1);
        gpu_reshape_nlist_kernel<0><<< grid, threads, 0, stream>>>(dest,
            N,
            NN,
            offset,
            batch_size,
            d_pos,
            box,
            d_n_neigh,
            d_nlist,
            d_head_list,
            rmax);
    }

    return cudaSuccess;

    }
