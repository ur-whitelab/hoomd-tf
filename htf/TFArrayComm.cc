// Copyright (c) 2020 HOOMD-TF Developers

#include "TFArrayComm.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace hoomd_tf
{
    //! Cast an int address to the corresponding pointer
    void *int2ptr(int64_t address) { return reinterpret_cast<void *>(address); }

    void export_TFArrayComm(pybind11::module &m)
    {
        pybind11::class_<TFArrayComm<TFCommMode::CPU, double>,
                         std::shared_ptr<TFArrayComm<TFCommMode::CPU, double>>>(
            m, "TFArrayCommCPU")
            .def(pybind11::init())
            .def("getArray", &TFArrayComm<TFCommMode::CPU, double>::getArray,
                 pybind11::return_value_policy::take_ownership);

        m.def("int2ptr", &int2ptr);
    }
} // namespace hoomd_tf
