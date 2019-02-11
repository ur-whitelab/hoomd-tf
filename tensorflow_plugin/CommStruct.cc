#include "CommStruct.h"

namespace hoomd_tf{

  template<>
  CommStructDerived<Scalar4>::CommStructDerived(GPUArray<Scalar4>& array, const char* name) :
    _array(&array),
    CommStruct(2, sizeof(Scalar), name){
      int tmp[] = {array.getNumElements(), 4};
      set_num_elements(tmp);
  }

  template<>
  CommStructDerived<Scalar>::CommStructDerived(GPUArray<Scalar>& array, const char* name) :
    _array(&array),
    CommStruct(2, sizeof(Scalar), name) {
      int tmp[] = {array.getNumElements(), 1};
      set_num_elements(tmp);
    }
}
