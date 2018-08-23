#include "../../IPCTaskLock.h"
#include <sys/mman.h>
#include <stdint.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

void export_Spinlock(pybind11::module& m) {
    pybind11::class_<IPCTaskLock>(m, "TaskLock")
    .def(pybind11::init())
    .def("start", &IPCTaskLock::start)
    .def("end", &IPCTaskLock::end)
    .def("await", &IPCTaskLock::await);
}


PYBIND11_PLUGIN(_spinlock) {
    pybind11::module m("_spinlock");
    export_Spinlock(m);
    return m.ptr();
}