
#ifndef CommStruct_H_H
#define CommStruct_H_H

#if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace hoomd_tf {
  struct CommStruct_t {

    CommStruct_t(void* mem_handle, const size_t* num_elements,
                size_t num_dims, size_t element_size,
                const char* name) :
      mem_handle(mem_handle),
      num_elements(num_elements),
      num_dims(num_dims),
      element_size(element_size),
      name(name) {
      size_t size = 1;
      for(unsigned int i; i < num_dims, i++)
        size *= num_elements[i]
      mem_size = size * _element_size;
    }

    CommStruct_t() {}

    void* mem_handle;
    const size_t* num_elements;
    size_t num_dims;
    size_t element_size;
    size_t mem_size;
    const char* name;
    //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
    #if defined(ENABLE_CUDA) || defined(GOOGLE_CUDA)
    cudaEvent_t event_handle;
    cudaStream_t stream = 0;
    #endif
  };
}
#endif