#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

class IpcTester  {

    public:
        IpcTester(pybind11::object& py_self);

        virtual void test();
        pybind11::object& _py_self;
};

void export_IpcTester(pybind11::module& m);