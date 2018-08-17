#ifndef _IPC_ARRAY_COMM_
#define _IPC_ARRAY_COMM_

#include <hoomd/GPUArray.h>
#include <hoomd/ExecutionConfiguration.h>
#include <string.h>
#include <stdexcept>
#include <iostream>
#include <sys/mman.h>


// I do not use specialization
// to treat the CommMode because you cannot do partial specialization of a method
// The overhead of the ifs is nothing, since the compiler will
// see them as if (1 == 0) and if(1 == 1) so they will be optimized.

enum class IPCCommMode{GPU, CPU};

struct IPCReservation{
    char* _ptr;
    size_t _index;
    size_t _size;

    IPCReservation() : _ptr(nullptr), _index(0), _size(0) {
    }
    IPCReservation(size_t size) : _ptr(nullptr), _index(0), _size(size) {
        _ptr = (char*) mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
            if(_ptr == MAP_FAILED)
                throw std::runtime_error("Unable to create mmap");
    }

    ~IPCReservation() {
        if(_ptr) {
            munmap((void*) _ptr, _size);
            _ptr = nullptr;
        }
    }

    void* allocate(size_t size) {
        if(size > (_size - _index))
            return nullptr;
        void* result =  _ptr + _index;
        _index += size;
        return result;

    }
};

// need this without access to context
#ifdef ENABLE_CUDA
void ipcCheckCudaError(cudaError_t err, const char *file, unsigned int line);
#define IPC_CHECK_CUDA_ERROR() { \
    cudaError_t err_sync = cudaGetLastError();  \
    ipcCheckCudaError(err_sync, __FILE__, __LINE__); \
    cudaError_t err_async = cudaDeviceSynchronize(); \
    ipcCheckCudaError(err_async, __FILE__, __LINE__); \
  }
#else
#define IPC_CHECK_CUDA_ERROR()
#endif //ENABLE_CUDA

// M: Communication mode (GPU or CPU)
// two-way communication where one owns array and other does not
//own_array: if it owns the array, it does not own the underlying data. Thus can use array as mapping
//!own_array: it has reference to underlying data.
template <IPCCommMode M, typename T> class IPCArrayComm {
    public:

        IPCArrayComm() :
            _mm_page(nullptr), _array(nullptr), _array_size(0), _own_array(false), _ipcr(nullptr)
        {
            checkDevice();
	    #ifdef ENABLE_CUDA
	    _ipc_array = nullptr;
	    _ipc_handle = nullptr;
	    #endif
        }

        IPCArrayComm(void* mm_page, size_t array_size, std::shared_ptr<const ExecutionConfiguration> exec_conf) :
            _mm_page(mm_page), _array(nullptr), _array_size(array_size), _own_array(true), _ipcr(nullptr)
        {
            checkDevice();
            _array = new GPUArray<T>(array_size, exec_conf);
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, IPCReservation* ipcr) :
            _mm_page(nullptr), _array(&gpu_array), _array_size(0), _own_array(false), _ipcr(ipcr)
        {
            checkDevice();
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, IPCReservation* ipcr, size_t num_elements) :
            _mm_page(nullptr), _array(&gpu_array), _array_size(0), _own_array(false), _ipcr(ipcr)
        {
  	    _array_size = num_elements * sizeof(T);
	    checkDevice();
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, void* mm_page) :
            _mm_page(mm_page), _array(&gpu_array), _array_size(0), _own_array(false), _ipcr(nullptr)
        {
            checkDevice();
            allocate();
        }


        IPCArrayComm(IPCArrayComm&& other)
        {
            //use the assignment operator
            *this = std::move(other);
        }
        IPCArrayComm& operator=(IPCArrayComm&& other) {
           checkDevice();
           //copy over variables
            _mm_page = other._mm_page;
            _array = other._array;
            _array_size = other._array_size;
	    other._array_size = 0;
            _own_array = other._own_array;
            //prevent mm_page from being deallocated in other
            other._mm_page = nullptr;
            //prevent other from deleting array
            other._own_array = false;

            #ifdef ENABLE_CUDA
            _ipc_array = other._ipc_array;
            other._ipc_array = nullptr;
            _ipc_handle = other._ipc_handle;
            other._ipc_handle = nullptr;
            #endif

            return *this;
        }

        ~IPCArrayComm() {
            if(_own_array)
                delete _array;
            this->deallocate();
        }

        void receive() {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<T> handle(*_array, access_location::host,access_mode::overwrite);
                memcpy(handle.data, _mm_page, getMMSize());
            } else {
                #ifdef ENABLE_CUDA
                ArrayHandle<T> handle(*_array, access_location::device, access_mode::overwrite);
                cudaMemcpy(handle.data, _ipc_array, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice);
                IPC_CHECK_CUDA_ERROR();
                #endif
            }
        }

        template <typename Func>
        void receiveOp(Func fun) {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<T> handle(*_array, access_location::host, access_mode::readwrite);
                //have to use funny fun.template because compiler thinks
                //that the '<' means less than instead of start of template arguments
                fun.template call<M>(handle.data, static_cast<T*> (_mm_page));
            } else {
                #ifdef ENABLE_CUDA
                ArrayHandle<T> handle(*_array, access_location::device, access_mode::readwrite);
                fun.template call<M>(handle.data, static_cast<T*> (_ipc_array));
                #endif
            }
        }

        void send() {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
                memcpy(_mm_page, handle.data, getMMSize());
            } else {
            #ifdef ENABLE_CUDA
                ArrayHandle<T> handle(*_array, access_location::device, access_mode::read);
                cudaMemcpy(_ipc_array, handle.data, getArraySize(), cudaMemcpyDeviceToDevice);
                IPC_CHECK_CUDA_ERROR();
            #endif
            }
        }

        std::vector<T> getArray() const {
            ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
	    return std::vector<T>(handle.data, handle.data + _array->getNumElements());

        }

        int64_t getAddress() const {
            return reinterpret_cast<int64_t> (_mm_page);
        }

    protected:

        void checkDevice() {
            #ifndef ENABLE_CUDA
            if(M == IPCCommMode::GPU)
                throw std::runtime_error("CUDA compilation not enabled so cannot use GPU CommMode");
            #endif
        }

        void allocate() {
            if(_array_size == 0)
                _array_size = sizeof(T) * _array->getNumElements();
            if(M == IPCCommMode::CPU) {
                if(!_mm_page) {
                    allocate_mm_page();
                }
            }
            #ifdef ENABLE_CUDA
            _ipc_array = nullptr;
            if(M == IPCCommMode::GPU) {
                //flush errors
                IPC_CHECK_CUDA_ERROR();
                if(_mm_page) {
                    //we will open the existing mapped memory in cuda
                    _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
                    cudaIpcOpenMemHandle(&_ipc_array, *_ipc_handle, cudaIpcMemLazyEnablePeerAccess);
                } else {
                    //We will create a shared block
                    allocate_mm_page();
                    _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
                    cudaMalloc((void**) &_ipc_array, getArraySize());
		    cudaIpcGetMemHandle(_ipc_handle, _ipc_array);
                }
                IPC_CHECK_CUDA_ERROR();
            }
            #endif
        }

        void allocate_mm_page() {
            if(_ipcr) {
                _mm_page = _ipcr->allocate(getMMSize());
                if(!_mm_page)
                    throw std::runtime_error("Unable to create mmap - IPCReservation not large enough");
            } else {
                _mm_page = mmap(nullptr, getMMSize(), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
                if(_mm_page == MAP_FAILED)
                    throw std::runtime_error("Unable to create mmap");
            }
        }

        void deallocate() {
            if(_mm_page) {
	        if(!_ipcr)
                  munmap(_mm_page, getMMSize());
            }
            if(M == IPCCommMode::GPU) {
                #ifdef ENABLE_CUDA
                if(_ipc_handle && _own_array)
                    cudaIpcCloseMemHandle(_ipc_handle);
                else if(_ipc_array)
		  cudaFree(_ipc_array);
                #endif
            }
        }

        size_t getArraySize() const {
            return _array_size;
        }

        size_t getMMSize() const {
            if(M == IPCCommMode::CPU)
                return getArraySize();
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
        size_t _array_size;
        bool _own_array;
        IPCReservation* _ipcr;
        #ifdef ENABLE_CUDA
        cudaIpcMemHandle_t* _ipc_handle;
        void*   _ipc_array;
        #endif
};

void export_IPCArrayComm(pybind11::module& m);

#endif //_IPC_ARRAY_COMM_
