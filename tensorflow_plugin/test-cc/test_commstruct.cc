#include "hoomd/tensorflow_plugin/CommStruct.h"

/*! \file test_commstruct.cc
    \brief Implements unit tests for the CommStruct object class
    \ingroup htf_unit_tests
*/

using namespace std;

#include "hoomd/test/up11_config.h"
HOOMD_UP_MAIN();

UP_TEST( CommStruct_basic_tests)
    {
    int dims = 1;
    size_t elem_size = sizeof(Scalar);
    const char* name = "test_commstruct";
    CommStruct test_struct(dims, elem_size, name);
    UP_ASSERT(test_struct.name == name);
    UP_ASSERT(test_struct.element_size == elem_size);
    // mem_size should default to 0 because it's empty
    UP_ASSERT(test_struct.mem_size == 0);
    }
