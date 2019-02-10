#include "CommStruct.h"

namespace hoomd_tf{

  template<>
  CommStructDerived<Scalar4>::CommStructDerived(GPUArray<Scalar4>& array, const char* name) :
    _array(&array),
    CommStruct({array.getNumElements(), 4}, 2, sizeof(Scalar), name){
  }

  template<>
  CommStructDerived<Scalar>::CommStructDerived(GPUArray<Scalar>& array, const char* name) :
    _array(&array),
    CommStruct({array.getNumElements(), 1}, 2, sizeof(Scalar), name) {
    }
}
