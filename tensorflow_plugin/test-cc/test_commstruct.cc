#include "hoomd/tensorflow_plugin/CommStruct.h"

/*! \file test_commstruct.cc
    \brief Implements unit tests for the CommStruct object class
    \ingroup htf_unit_tests
*/

using namespace std;

#include "hoomd/test/up11_config.h"
HOOMD_UP_MAIN();

//#define UP_ASSERT_CLOSE(a,b,eps)					\
//#define UP_ASSERT_SMALL(a,eps)					\
//#define CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
//#define CHECK_SMALL(a,c) UP_ASSERT(std::abs(a) < c)
//#define MY_CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
//#define MY_CHECK_SMALL(a,c) CHECK_SMALL( a, Scalar(c))
//#define CHECK_EQUAL_UINT(a,b) UP_ASSERT_EQUAL(a,(unsigned int)(b))
//#define MY_ASSERT_EQUAL(a,b) UP_ASSERT(a == b)
//const Scalar tol_small = Scalar(1e-3);
//const Scalar tol = Scalar(1e-2);
//! Loose tolerance to be used with randomly generated and unpredictable comparisons
//Scalar loose_tol = Scalar(10);

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
