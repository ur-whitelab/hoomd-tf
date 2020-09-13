// Copyright (c) 2020 HOOMD-TF Developers

// Include the defined classes that are to be exported to python
#include "TensorflowCompute.h"
#include "TFArrayComm.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hoomd_tf;

/*! \file module.cc
 *  Specifies the python module. Note that the name must expliclty match the
 *  PROJECT() name provided in CMakeLists (with an underscore in front)
 */

PYBIND11_PLUGIN(_htf)
    {
    pybind11::module m("_htf");
    export_TensorflowCompute(m);

    #ifdef ENABLE_CUDA
        export_TensorflowComputeGPU(m);
    #endif

    export_TFArrayComm(m);

    return m.ptr();
    }
