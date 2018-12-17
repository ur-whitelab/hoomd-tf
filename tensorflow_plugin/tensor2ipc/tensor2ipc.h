#ifndef TENSOR2IPC
#define TENSOR2IPC

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

// need to make sure the hoomd-specific cuda flags get set for include
#ifdef GOOGLE_CUDA
#define ENABLE_CUDA
#endif

#include "../IPCStruct.h"

using namespace tensorflow;

template <typename Device, typename T>
struct TF2IPCFunctor {
  void operator()(const Device& d, int size, IPCStruct_t* address, const T* in
                  );
};

#ifdef GOOGLE_CUDA
template <typename T>
struct TF2IPCFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, IPCStruct_t* address,
                  const T* in);
};
#endif

#endif  // TENSOR2IPC
