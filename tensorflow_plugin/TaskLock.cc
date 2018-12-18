#include "TaskLock.h"

using namespace hoomd_tf;

TaskLock* make_tasklock() { return new TaskLock(); }

void hoomd_tf::export_TaskLock(pybind11::module& m) {
pybind11::class_<TaskLock>(m, "TaskLock")
    .def(pybind11::init())
    .def("start", &TaskLock::start)
    .def("end", &TaskLock::end)
    .def("exit", &TaskLock::exit)
    .def("is_exit", &TaskLock::is_exit)
    .def("await", &TaskLock::await);

m.def("make_tasklock", &make_tasklock,
        pybind11::return_value_policy::reference);
}
