#ifndef IPC2TENSOR
#define IPC2TENSOR

// I don't know which is needed for int64 and I'm lazy
//TODO: figure
#include "tensorflow/core/framework/op.h"


template <typename Device, typename T>
struct IPC2TFunctor {
  void operator()(const Device& d, int size, tensorflow::int64 address, T** ipc_memory, T* out);
};


#endif //IPC2TENSOR