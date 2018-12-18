


struct IPCStruct_t {
  void* mem_handle;
  size_t num_elements;
  size_t element_size;
  //TODO Why is ENABLE_CUDA set for compilng tf code? We don't have any hoomd headers...
  #if (defined(ENABLE_CUDA) && !defined(TF_COMPILE)) || defined(GOOGLE_CUDA)
  cudaEvent_t event_handle;
  cudaStream_t stream = 0;
  #endif
};
