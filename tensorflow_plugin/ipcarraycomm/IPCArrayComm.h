#ifndef _IPC_ARRAY_COMM_
#define _IPC_ARRAY_COMM_

#include <hoomd/GPUArray.h>
#include <string.h>
#include <stdexcept>
#include <sys/mman.h>


enum class IPCCommMode{GPU, CPU};

// M: Communication mode
template <IPCCommMode M, typename T> class IPCArrayComm {
    public:

        IPCArrayComm(GPUArray<T>& gpu_array) :
            _mm_page(NULL), _array(&gpu_array), _own_array(false)
        {
            checkDevice();
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, void* mm_page) :
            _mm_page(mm_page), _array(&gpu_array), _own_array(false)
        {
            checkDevice();
            allocate();
        }


        IPCArrayComm(const IPCArrayComm& other) :
            _mm_page(NULL), _array(NULL), _own_array(true)
        {
            checkDevice();
            _mm_page = other._mm_page;
            _array = new GPUArray<T>(other._array);
            allocate();
        }

        ~IPCArrayComm() {
            if(_own_array)
                delete _array;
            deallocate();
        }


        void deallocate() {
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

        void update_array(GPUArray<T>& array) {
            if(!_own_array && _array != array) {
                if(array.getNumElements() != _array->getNumElements()) {
                    dellocate();
                    allocate();
                }
                _array = &array;
            }
        }

        void receive() {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<Scalar4> handle(*_array, access_location::host);
                memcpy(handle.data, _mm_page, get_mm_size());
            } else {
                #ifdef ENABLE_CUDA
                ArrayHandle<Scalar4> handle(*_array, access_location::device);
                cudaMemcpy(handle.data, _ipc_array, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice);
                #endif
            }
        }

        void send() {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<Scalar4> handle(*_array, access_location::host);
                memcpy(_mm_page, handle.data, get_mm_size());
            } else {
            #ifdef ENABLE_CUDA
                ArrayHandle<Scalar4> handle(*_array, access_location::device);
                cudaMemcpy(_ipc_array, handle.data, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice);
            #endif
            }
        }

        GPUArray<T>& getArray() {
            return *_array;
        }

    protected:

        void checkDevice() {
            #ifndef ENABLE_CUDA
            if(M == IPCCommMode::GPU)
                throw std::runtime_error("CUDA compilation not enabled so cannot use GPU CommMode");
            #endif
        }

        void allocate() {
            if(!_own_array) {
                _mm_page = mmap(NULL, get_mm_size(), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
                if(_mm_page == MAP_FAILED)
                    throw std::runtime_error("Unable to create mmap");
            }
            #ifdef ENABLE_CUDA
            if(_own_array) {
                _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
                cudaIpcOpenMemHandle(&_ipc_array, *_ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            } else {
                ArrayHandle<Scalar4> handle(*_array, access_location::device);
                cudaIpcGetMemHandle(_ipc_handle, static_cast<void*> (handle.data));
                _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
            }
            #endif
        }

        size_t get_array_size() const {
            return sizeof(T) * _array->getNumElements();
        }

        size_t get_mm_size() const {
            if(M == IPCCommMode::CPU)
                return get_array_size();
            else {
                #ifdef ENABLE_CUDA
                return sizeof(cudaIpcMemHandle_t);
                #endif
            }
            return 0;
        }

        void* _mm_page;
    private:
        GPUArray<T>* _array;
        bool _own_array;
        #ifdef ENABLE_CUDA
        cudaIpcMemHandle_t* _ipc_handle;
        void*   _ipc_array;
        #endif
};

#endif //_IPC_ARRAY_COMM_