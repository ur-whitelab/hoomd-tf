#ifndef _IPC_ARRAY_COMM_
#define _IPC_ARRAY_COMM_

#include <hoomd/GPUArray.h>
#include <string.h>
#include <stdexcept>
#include <sys/mman.h>


enum class IPCCommMode{GPU, CPU};

// M: Communication mode (GPU or CPU)
// two-way communication where one owns array and other does not
//own_array: if it owns the array, it does not own the underlying data. Thus can use array as mapping
//!own_array: it has reference to underlying data.
template <IPCCommMode M, typename T> class IPCArrayComm {
    public:

        IPCArrayComm() :
            _mm_page(NULL), _array(NULL), _array_size(0), _own_array(false)
        {
            checkDevice();
        }

        IPCArrayComm(void* mm_page, size_t array_size, std::shared_ptr<const ExecutionConfiguration> exec_conf) :
            _mm_page(mm_page), _array(NULL), _array_size(array_size), _own_array(true)
        {
            checkDevice();
            _array = new GPUArray<T>(array_size, exec_conf);
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array) :
            _mm_page(NULL), _array(&gpu_array), _array_size(0), _own_array(false)
        {
            checkDevice();
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, size_t array_size) :
            _mm_page(NULL), _array(&gpu_array), _array_size(array_size), _own_array(false)
        {
            checkDevice();
            allocate();
        }

        IPCArrayComm(GPUArray<T>& gpu_array, void* mm_page) :
            _mm_page(mm_page), _array(&gpu_array), _array_size(0), _own_array(false)
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
            _own_array = other._own_array;
            //prevent mm_page from being deallocated in other
            other._mm_page = NULL;
            //prevent other from deleting array
            other._own_array = false;

            #ifdef ENABLE_CUDA
            _ipc_array = other._ipc_array;
            _ipc_handle = other._ipc_handle;
            #endif

            return *this;
        }

        ~IPCArrayComm() {
            if(_own_array)
                delete _array;
            this->deallocate();
        }

        void update_array(GPUArray<T>& array) {
            if(!_own_array && _array != array) {
                if(array.getNumElements() != _array->getNumElements()) {
                    this->dellocate();
                    allocate();
                }
                _array = &array;
            }
        }

        void receive() {
            if(M == IPCCommMode::CPU) {
                ArrayHandle<T> handle(*_array, access_location::host,access_mode::overwrite);
                memcpy(handle.data, _mm_page, getMMSize());
            } else {
                #ifdef ENABLE_CUDA
                ArrayHandle<T> handle(*_array, access_location::device, access_mode::overwrite);
                cudaMemcpy(handle.data, _ipc_array, _array->getNumElements() * sizeof(T),  cudaMemcpyDeviceToDevice);
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
                cudaMemcpy(_ipc_array, handle.data, getArraySize(),  cudaMemcpyDeviceToDevice);
            #endif
            }
        }

        std::vector<T> getArray() const {
            ArrayHandle<T> handle(*_array, access_location::host, access_mode::read);
            std::vector<T> array(handle.data, handle.data + getArraySize());
            return array;
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
            if(!_mm_page) {
                _mm_page = mmap(NULL, getMMSize(), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
                std::cout << "Allocating" << _mm_page << std::endl;
                if(_mm_page == MAP_FAILED)
                    throw std::runtime_error("Unable to create mmap");
            }
            #ifdef ENABLE_CUDA
            if(_mm_page) {
                //we will open the existing mapped memory in cuda
                _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
                cudaIpcOpenMemHandle(&_ipc_array, *_ipc_handle, cudaIpcMemLazyEnablePeerAccess);
            } else {
                //We will create a shared block
                _ipc_handle = reinterpret_cast<cudaIpcMemHandle_t*> (_mm_page);
                cudaMalloc(&_ipc_array, getArraySize());
                cudaIpcGetMemHandle(_ipc_handle, _ipc_array);

            }
            #endif
        }

        void deallocate() {
            if(_mm_page) {
                std::cout << "Deallocating" << _mm_page << std::endl;
                munmap(_mm_page, getMMSize());
            }
            if(M == IPCCommMode::GPU) {
                #ifdef ENABLE_CUDA
                if(_ipc_array && _own_array)
                    cudaIpcCloseMemHandle(_ipc_array);
                else if(_ipc_array && !_own_array)
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
        #ifdef ENABLE_CUDA
        cudaIpcMemHandle_t* _ipc_handle;
        void*   _ipc_array;
        #endif
};

#endif //_IPC_ARRAY_COMM_