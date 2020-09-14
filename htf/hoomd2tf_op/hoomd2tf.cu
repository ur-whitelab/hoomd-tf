// Copyright (c) 2020 HOOMD-TF Developers

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "hoomd2tf.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// GPU specialization of actual computation.
template <typename T>
void HOOMD2TFFunctor<GPUDevice, T >::operator()(
    const GPUDevice& d, int size, CommStruct* in_memory,T* out) {

  //cudaEventSynchronize(in_memory.event);
  in_memory->readGPUMemory(out, size * sizeof(T));
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct HOOMD2TFFunctor<GPUDevice, float> ;
template struct HOOMD2TFFunctor<GPUDevice, double> ;

#endif  // GOOGLE_CUDA
