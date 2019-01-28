#include "CommStruct.h"

using namespace hoomd_tf;

  template<>
  struct CommStructDerived<Scalar4> {

  CommStructDerived() : array(GPUArray<Scalar4>(1))
    {

    }

    CommStructDerived(GPUArray<Scalar4>& array, const char* name) :
    CommStruct({array->getNumElements(), 4}, 2, sizeof(Scalar), name),
    array(array) {

  };

  template<>
  struct CommStructDerived<Scalar > {

  CommStructDerived() : array(GPUArray<Scalar>(1))
    {

    }

    CommStructDerived(GPUArray<Scalar>& array, const char* name) :
    CommStruct({array->getNumElements(), 1}, 2, sizeof(Scalar), name),
    array(array) {

  };
