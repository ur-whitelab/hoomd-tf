#ifndef IPC2TENSOR_H_
#define IPC2TENSOR_H_

#include "tensorflow/core/framework/types.h"


template <typename Device, typename T>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, void* address, T** ipc_memory, T* out);
};

#ifdef GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct IPC2TFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, void* address, T** ipc_memory, T* out);
  };
#endif


#endif //IPC2TENSOR_H_ 
