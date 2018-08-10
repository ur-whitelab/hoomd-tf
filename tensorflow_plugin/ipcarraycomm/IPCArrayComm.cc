
IPCArrayComm::IPCArrayComm(const GPUArray<T>& gpu_array) :
    _mm_page(NULL), _array(gpu_array), _own_array(false)
{
    checkDevice<MT();
    allocate<M,T();
}

IPCArrayComm::IPCArrayComm(const GPUArray<T>& gpu_array, void* mm_page) :
    _mm_page(mm_page), _array(gpu_array), _own_array(false)
{
    checkDevice();
    allocate();
}


IPCArrayComm::IPCArrayComm(const IPCArrayComm& other) :
    _mm_page(NULL), _array(NULL), _own_array(true)
{
    checkDevice();
    _mm_page = other._mm_page;
    _array = new GPUArray(other._array);
    allocate();
}

IPCArrayComm::~IPCArrayComm() {
    if(_own_array)
        delete _array;
    if(_mm_page) {
        munmap(_mm_page, get_mm_size());
    }
    if(M == IPCCommMode::GPU) {
        #ifdef ENABLE_CUDA
        if(_ipc_array)
            cudaIpcCloseMemHandle(_ipc_array);
        #endif
    }
}

void IPCArrayComm::checkDevice() {
    #ifndef ENABLE_CUDA
    if(M == IPCCommMode::GPU)
        throw std::runtime_error("CUDA compilation not enabled so cannot use GPU CommMode");
    #endif
}

void IPCArrayComm::receive() {
    if(M == IPCCommMode::CPU) {
        ArrayHandle<Scalar4> handle(_array, access_location::host);
        memcpy(handle.data, _mm_page, get_mm_size());
    } else {
        #ifdef ENABLE_CUDA
        ArrayHandle<Scalar4> handle(_array, access_location::device);
        cudaMemcpy(handle.data, _ipc_array, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice)
        #endif
    }
}

void IPCArrayComm::send() {
    if(M == IPCCommMode::CPU) {
        ArrayHandle<Scalar4> handle(_array, access_location::host);
        memcpy(_mm_page, handle.data, get_mm_size());
    } else {
    #ifdef ENABLE_CUDA
        ArrayHandle<Scalar4> handle(_array, access_location::device);
        cudaMemcpy(_ipc_array, handle.data, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice)
    #endif
    }
}

void IPCArrayComm::allocate() {
    if(!_own_array) {
        if(M == IPCCommMode::CPU)
            _mm_page = mmap(NULL, get_mm_size(), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
        else {
            #ifdef ENABLE_CUDA
            _mm_page = mmap(NULL, sizeof(cudaIpcMemHandle_t), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0));
            #endif
        }
    }
    #ifdef ENABLE_CUDA
    if(_own_array) {
        _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> _mm_page;
        cudaIpcOpenMemHandle(&_ipc_array, _ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    } else {
        ArrayHandle<Scalar4> handle(_array, access_location::device);
        cudaIpcGetMemHandle(_ipc_handle, static_cast<void*> handle.data)
        _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t> _mm_page;
    }
    #endif
}


void IPCArrayComm::get_array_size() const {
    return sizeof(T) * _array->getNumElements();
}