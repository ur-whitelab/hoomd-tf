
#ifndef CommStruct_H_H
#define CommStruct_H_H

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace hoomd_tf {
  struct CommStruct_t {
    void* mem_handle;
    size_t num_elements;
    size_t element_size;
    //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
    #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
    cudaEvent_t event_handle;
    cudaStream_t stream = 0;
    #endif
  };
}
#endif