#ifndef TF2HOOMD
#define TF2HOOMD

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../CommStruct.h"

using namespace tensorflow;
using namespace hoomd_tf;

template <typename Device, typename T>
struct TF2IPCFunctor {
  void operator()(const Device& d, int size, CommStruct* address, const T* in
                  );
};

#ifdef GOOGLE_CUDA
template <typename T>
struct TF2IPCFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, CommStruct* address,
                  const T* in);
};
#endif

#endif  // TF2HOOMD
