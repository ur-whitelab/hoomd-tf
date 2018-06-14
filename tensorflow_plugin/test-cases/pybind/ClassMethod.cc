#include "ClassMethod.h"

ClassMethod::ClassMethod(pybind11::object& py_self): _py_self(py_self) {}

void ClassMethod::test() {
    _py_self.attr("test")();
}

//doing this two-step approach to mimic what's done by hoomd
PYBIND11_PLUGIN(_class_method) {
    pybind11::module m("_class_method");
    export_ClassMethod(m);
    return m.ptr();
}
void export_ClassMethod(pybind11::module& m) {
    pybind11::class_<ClassMethod, std::shared_ptr<ClassMethod> >(m, "ClassMethod")
    .def(pybind11::init<pybind11::object &>())
    .def("test", &ClassMethod::test);
}