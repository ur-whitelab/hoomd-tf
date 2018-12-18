#ifndef IPC2TENSOR_H_
#define IPC2TENSOR_H_

#include "tensorflow/core/framework/types.h"
#include "../IPCStruct.h"

template <typename Device, typename T>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, IPCStruct_t* address,
                  T* out);
};

#ifdef GOOGLE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

// Partially specialize functor for GpuDevice.
template <typename T>
struct IPC2TFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, IPCStruct_t* address,
                  T* out);
};
#endif

#endif  // IPC2TENSOR_H_
