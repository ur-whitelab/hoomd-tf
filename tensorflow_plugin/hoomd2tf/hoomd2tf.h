#ifndef HOOMD2TF_H_
#define HOOMD2TF_H_

#include "tensorflow/core/framework/types.h"
#include "../CommStruct.h"

using namespace hoomd_tf;

template <typename Device, typename T>
struct HOOMD2TFFunctor {
  void operator()(const Device& d, int size, CommStruct_t* address,
                  T* out);
};

#ifdef GOOGLE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

// Partially specialize functor for GpuDevice.
template <typename T>
struct HOOMD2TFFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, CommStruct_t* address,
                  T* out);
};
#endif

#endif  // HOOMD2TF_H_
