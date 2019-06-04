#include <hoomd/ExecutionConfiguration.h>
#include <hoomd/htf/CommStruct.h>

/*! \file test_commstruct.cc
    \brief Implements unit tests for the CommStruct object class
    \ingroup htf_unit_tests
*/

using namespace std;

#include <hoomd/test/upp11_config.h>
HOOMD_UP_MAIN();

UP_TEST( CommStruct_basic_tests)
    {
    unsigned int n_elems = 1;
    const ExecutionConfiguration *exec_conf = new ExecutionConfiguration();
    std::shared_ptr<const ExecutionConfiguration> conf_ptr;
    const std::string tag = "test";
    GlobalArray<Scalar> test_array(n_elems, conf_ptr, tag);
    size_t elem_size = sizeof(Scalar);
    const char* name = "test_commstruct";
    hoomd_tf::CommStructDerived<Scalar> test_struct(test_array, name);
    UP_ASSERT(test_struct.name == name);
    UP_ASSERT(test_struct.element_size == elem_size);
    // mem_size should default to 0 because it's empty
    UP_ASSERT(test_struct.mem_size == 0);
    }
