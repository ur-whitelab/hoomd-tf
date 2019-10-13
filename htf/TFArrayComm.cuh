// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#ifndef m_TF_ARRAY_COMM_CUH_
#define m_TF_ARRAY_COMM_CUH_

#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>
#include <hoomd/Index1D.h>
#include <hoomd/HOOMDMath.h>


/*! \file TFArrayComm.cuh
 *  \brief Declarations of GPU version of some TFArrayComm methods
 */ 

//! Unstuff integers in gpu array
extern "C" cudaError_t htf_gpu_unstuff4(Scalar4 *arrray,
                                       unsigned int N,
                                       cudaStream_t stream);

#endif // m_TF_ARRAY_COMM_CUH_