// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

// Include the defined classes that are to be exported to python
#include "TensorflowCompute.h"
#include "TFArrayComm.h"
#include "TaskLock.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace hoomd_tf;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_PLUGIN(_tensorflow_plugin)
    {
    pybind11::module m("_tensorflow_plugin");
    export_TensorflowCompute(m);

    #ifdef ENABLE_CUDA
    export_TensorflowComputeGPU(m);
    #endif

    export_TFArrayComm(m);

    export_TaskLock(m);

    return m.ptr();
    }
