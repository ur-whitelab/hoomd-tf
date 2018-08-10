#include "IpcTester.h"

IpcTester::IpcTester(pybind11::object& py_self, size_t length): _py_self(py_self), _length(length) {
    _input_buffer = new Scalar4[length];
    _output_buffer = new Scalar4[length];

    for(unsigned int i = 0; i < length; i++) {
        _input_buffer[i].x = 1;
        _input_buffer[i].y = 1;
        _input_buffer[i].z = 1;
        _input_buffer[i].w = 1;
        _output_buffer[i].x = 0;
        _output_buffer[i].y = 0;
        _output_buffer[i].z = 0;
        _output_buffer[i].w = 0;
    }

}

IpcTester::~IpcTester() {
    delete _input_buffer;
    delete _output_buffer;
    //
}



std::vector<Scalar4> IpcTester::get_input_array() const {
    std::vector<Scalar4> array(_input_buffer, _input_buffer + _length);
    return array;
}

std::vector<Scalar4> IpcTester::get_output_array() const {
    std::vector<Scalar4> array(_output_buffer, _output_buffer + _length);
    return array;
}

//doing this two-step approach to mimic what's done by hoomd
PYBIND11_PLUGIN(_ipc_tester) {
    pybind11::module m("_ipc_tester");
    export_IpcTester(m);

    return m.ptr();
}
void export_IpcTester(pybind11::module& m) {
    pybind11::class_<IpcTester, std::shared_ptr<IpcTester> >(m, "IpcTester")
    .def(pybind11::init<pybind11::object &, size_t>())
    .def("get_input_buffer", &IpcTester::get_input_buffer)
    .def("get_output_buffer", &IpcTester::get_output_buffer)
    .def("get_input_array", &IpcTester::get_input_array, pybind11::return_value_policy::take_ownership)
    .def("get_output_array", &IpcTester::get_output_array, pybind11::return_value_policy::take_ownership);

    pybind11::class_<double4, std::shared_ptr<double4> >(m,"double4")
    .def(pybind11::init<>())
    .def_readwrite("x", &double4::x)
    .def_readwrite("y", &double4::y)
    .def_readwrite("z", &double4::z)
    .def_readwrite("w", &double4::w);
}