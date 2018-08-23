#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <iostream>
#include "ipc2tensor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

struct InputMem_t {
  cudaIpcMemHandle_t mem_handle;
  cudaIpcEventHandle_t event_handle;
};

// GPU specialization of actual computation.
template <typename T>
void IPC2TFunctor<GPUDevice, T, cudaIPC_t<T> >::operator()(
    const GPUDevice& d, int size, void* input, cudaIPC_t<T>& ipc_memory,
    T* out) {
  if (!(ipc_memory.array)) {
    // TODO: Learn TF way to handle cuda errors
    auto ipc_handle = reinterpret_cast<InputMem_t*>(input);
    cudaIpcOpenMemHandle((void**)(&(ipc_memory.array)), ipc_handle->mem_handle,
                         cudaIpcMemLazyEnablePeerAccess);
    cudaIpcOpenEventHandle(&(ipc_memory.event), ipc_handle->event_handle);
  }
  cudaEventSynchronize(ipc_memory.event);
  cudaMemcpy((void*)(out), (const void*)(ipc_memory.array), size * sizeof(T),
             cudaMemcpyDeviceToDevice);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct IPC2TFunctor<GPUDevice, float, cudaIPC_t<float> >;
template struct IPC2TFunctor<GPUDevice, double, cudaIPC_t<double> >;

#endif  // GOOGLE_CUDA
