#include "IPCTaskLock.h"

IPCTaskLock* make_tasklock() {
  return new IPCTaskLock();
}

void export_IPCTaskLock(pybind11::module& m) {
    pybind11::class_<IPCTaskLock>(m, "IPCTaskLock")
    .def(pybind11::init())
    .def("start", &IPCTaskLock::start)
    .def("end", &IPCTaskLock::end)
    .def("exit", &IPCTaskLock::exit)
    .def("await", &IPCTaskLock::await);

    m.def("make_tasklock", &make_tasklock, pybind11::return_value_policy::reference);
}
