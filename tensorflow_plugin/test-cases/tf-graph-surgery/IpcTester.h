#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/HOOMDMath.h>


class IpcTester  {

    public:
        IpcTester(pybind11::object& py_self, size_t length);
        virtual ~IpcTester();

        std::vector<Scalar4> get_input_array() const;
        std::vector<Scalar4> get_output_array() const;


        int64_t get_input_buffer() const { return reinterpret_cast<int64_t> (_input_buffer);}
        int64_t get_output_buffer() const {return reinterpret_cast<int64_t> (_output_buffer);}

        pybind11::object& _py_self;
    private:
        const size_t _length;
        Scalar4* _input_buffer;
        Scalar4* _output_buffer;

};

void export_IpcTester(pybind11::module& m);