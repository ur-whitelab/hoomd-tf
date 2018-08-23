#ifndef IPC2TENSOR_H_
#define IPC2TENSOR_H_

#include "tensorflow/core/framework/types.h"

template <typename Device, typename T, typename IPCM>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, void* address, IPCM& ipc_memory,
                  T* out);
};

#ifdef GOOGLE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
struct cudaIPC_t {
  T* array;
  cudaEvent_t event;
  cudaIPC_t() : array(nullptr) {}
  ~cudaIPC_t() {
    cudaEventDestroy(event);
    if (array) cudaIpcCloseMemHandle(array);
  }
};

// Partially specialize functor for GpuDevice.
template <typename T>
struct IPC2TFunctor<Eigen::GpuDevice, T, cudaIPC_t<T> > {
  void operator()(const Eigen::GpuDevice& d, int size, void* address,
                  cudaIPC_t<T>& ipc_memory, T* out);
};
#endif

#endif  // IPC2TENSOR_H_
