// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "tf2hoomd.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
// TODO Add cuda error checks here
template <typename T>
void TF2IPCFunctor<GPUDevice, T>::operator()(const GPUDevice& d, int size,
    CommStruct* handle, const T* in) {
  handle->write_gpu_memory(in, size * sizeof(T));
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct TF2IPCFunctor<GPUDevice, float>;
template struct TF2IPCFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
