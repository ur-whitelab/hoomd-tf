


struct IPCStruct_t {
  void* mem_handle;
  size_t num_elements;
  size_t element_size;
  #ifdef ENABLE_CUDA
  cudaEvent_t event_handle;
  cudaStream_t stream = 0;
  #endif
};
