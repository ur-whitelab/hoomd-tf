// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#ifndef _TENSORFLOW_COMPUTE_CUH_
#define _TENSORFLOW_COMPUTE_CUH_

#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>
#include <hoomd/Index1D.h>
#include <hoomd/HOOMDMath.h>

/*! \file TensorflowCompute.cuh
 *  \brief Declarations of GPU version of some TensorflowCompute methods
 */ 

//! Add Scalar4 type arrays in parallel on GPU by invoking the kernel.
extern "C" cudaError_t gpu_add_scalar4(Scalar4 *dest,
                                       Scalar4 *src,
                                       unsigned int N,
                                       cudaStream_t stream);

//! Add up virial arrays in parallel on GPU by invoking the kernel.
extern "C" cudaError_t gpu_add_virial(Scalar *dest,
                                      Scalar *src,
                                      unsigned int N,
                                      unsigned int pitch,
                                      cudaStream_t stream);

/*! Reshape neighbor list to fit into GPU memory in an agreeable way for TensorFlow.
 *  Attempts to use texture memory first, where possible, or will
 *  use device main memory if not. Invokes kernel function.
 */
extern "C" cudaError_t gpu_reshape_nlist(Scalar4* dest,
                                         const Scalar4 *d_pos,
                                         const unsigned int N,
                                         const unsigned int NN,
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
					 cudaStream_t stream);

#endif // _TENSORFLOW_COMPUTE_CUH_
