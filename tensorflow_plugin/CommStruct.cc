#include "CommStruct.h"

using namespace hoomd_tf;

  template<>
  struct CommStructDerived<Scalar4> {

    CommStructDerived(GPUArray<Scalar4>& array, const char* name) :
    CommStruct({array->getNumElements(), 4}, 2, sizeof(Scalar), name){
        array.swap(array);
  };

  template<>
  struct CommStructDerived<Scalar > {
    CommStructDerived(GPUArray<Scalar>& array, const char* name) :
    CommStruct({array->getNumElements(), 1}, 2, sizeof(Scalar), name) {
        _array.swap(array);
  };
