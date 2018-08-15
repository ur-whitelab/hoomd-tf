#ifndef TENSOR2IPC
#define TENSOR2IPC

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;



template <typename Device, typename T>
struct TF2IPCFunctor {
  void operator()(const Device& d, int size, int64 address, const T* in, T** ipc_memory);
};

#endif //TENSOR2IPC