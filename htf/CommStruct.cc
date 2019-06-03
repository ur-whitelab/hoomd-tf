// Copyright (c) 2018 Andrew White at the University of Rochester
//  This file is part of the Hoomd-Tensorflow plugin developed by Andrew White

#include "CommStruct.h"

namespace hoomd_tf
    {
    /*! \file CommStruct.cc
      \brief CommStructDerived constructor definition
    */



    template<>
    CommStructDerived<Scalar4>::CommStructDerived(GlobalArray<Scalar4>& array, const char* name) :
        _array(&array),
        CommStruct(2, sizeof(Scalar), name)
        {
	int tmp[] = {static_cast<int>(array.getNumElements()), 4};
        setNumElements(tmp);
        }

    //! Define template constructor for Scalar dtype
    template<>
    CommStructDerived<Scalar>::CommStructDerived(GlobalArray<Scalar>& array, const char* name) :
        _array(&array),
        CommStruct(2, sizeof(Scalar), name)
        {
	  int tmp[] = {static_cast<int>(array.getNumElements()), 1};
        setNumElements(tmp);
        }
    }
