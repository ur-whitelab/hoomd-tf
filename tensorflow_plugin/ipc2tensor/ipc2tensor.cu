#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "ipc2tensor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

struct InputMem_t {
  void* mem_handle;
  cudaEvent_t event_handle;
  cudaStream_t stream;
};

// GPU specialization of actual computation.
template <typename T>
void IPC2TFunctor<GPUDevice, T, cudaIPC_t<T> >::operator()(
    const GPUDevice& d, int size, void* input, cudaIPC_t<T>& ipc_memory,
    T* out) {
  if (!(ipc_memory.array)) {
    // TODO: Learn TF way to handle cuda errors
    auto ipc_handle = reinterpret_cast<InputMem_t*>(input);
    ipc_memory.array = static_cast<T*>(ipc_handle->mem_handle);
  }
  //cudaEventSynchronize(ipc_memory.event);
  cudaMemcpy((void*)(out), (const void*)(ipc_memory.array), size * sizeof(T),
             cudaMemcpyDeviceToDevice);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct IPC2TFunctor<GPUDevice, float, cudaIPC_t<float> >;
template struct IPC2TFunctor<GPUDevice, double, cudaIPC_t<double> >;

#endif  // GOOGLE_CUDA
