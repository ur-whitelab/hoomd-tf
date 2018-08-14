// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _TENSORFLOW_COMPUTE_CUH_
#define _TENSORFLOW_COMPUTE_CUH_

#include <hoomd/HOOMDMath.h>

extern "C" cudaError_t gpu_add_scalar4(Scalar4 *dest, Scalar4 *src, unsigned int N);
extern "C" cudaError_t gpu_add_virial(Scalar *dest, Scalar *src, unsigned int N, unsigned int pitch);


#endif // _TENSORFLOW_COMPUTE_CUH_
