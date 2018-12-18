#include "../../TaskLock.h"
#include <sys/mman.h>
#include <stdint.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hoomd_tf;

void export_Spinlock(pybind11::module& m) {
    pybind11::class_<TaskLock>(m, "TaskLock")
    .def(pybind11::init())
    .def("start", &TaskLock::start)
    .def("end", &TaskLock::end)
    .def("await", &TaskLock::await);
}


PYBIND11_PLUGIN(_spinlock) {
    pybind11::module m("_spinlock");
    export_Spinlock(m);
    return m.ptr();
}