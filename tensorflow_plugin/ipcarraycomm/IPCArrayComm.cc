#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/stl.h>
#include <hoomd/extern/pybind/include/pybind11/stl_bind.h>
#include "IPCArrayComm.h"

void* int2ptr(int64_t address) {
    return reinterpret_cast<void*> (address);
}

void export_IPCArrayComm(pybind11::module& m) {
    pybind11::class_<IPCArrayComm<IPCCommMode::CPU, double>,  std::shared_ptr<IPCArrayComm<IPCCommMode::CPU, double> > >(m, "IPCArrayCommCPU")
    .def(pybind11::init<void*, size_t, std::shared_ptr<const ExecutionConfiguration> >())
    .def("getArray", &IPCArrayComm<IPCCommMode::CPU, double>::getArray, pybind11::return_value_policy::take_ownership)
    .def("send", &IPCArrayComm<IPCCommMode::CPU, double>::send)
    .def("receive", &IPCArrayComm<IPCCommMode::CPU, double>::receive)
    ;

    m.def("int2ptr", &int2ptr);
}

//doing this two-step approach to mimic what's done by hoomd
PYBIND11_PLUGIN(_ipc_array_comm) {
    pybind11::module m("_ipc_array_comm");
    export_IPCArrayComm(m);
    return m.ptr();
}