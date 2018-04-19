// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _TENSORFLOW_UPDATER_CUH_
#define _TENSORFLOW_UPDATER_CUH_

// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file TensorflowUpdater.cuh
    \brief Declaration of CUDA kernels for TensorflowUpdater
*/

// A C API call to run a CUDA kernel is needed for TensorflowUpdaterGPU to call
//! Zeros velocities on the GPU
extern "C" cudaError_t gpu_zero_velocities(Scalar4 *d_vel, unsigned int N);

#endif // _TENSORFLOW_UPDATER_CUH_
