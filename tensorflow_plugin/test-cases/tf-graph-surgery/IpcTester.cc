#include "IpcTester.h"

IpcTester::IpcTester(pybind11::object& py_self): _py_self(py_self) {
    _py_self.attr("test")();
}

void IpcTester::test() {
    _py_self.attr("test")();
}

//doing this two-step approach to mimic what's done by hoomd
PYBIND11_PLUGIN(_class_method) {
    pybind11::module m("_class_method");
    export_IpcTester(m);
    return m.ptr();
}
void export_IpcTester(pybind11::module& m) {
    pybind11::class_<IpcTester, std::shared_ptr<IpcTester> >(m, "IpcTester")
    .def(pybind11::init<pybind11::object &>())
    .def("test", &IpcTester::test);
}