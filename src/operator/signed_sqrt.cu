/*!
 * Copyright (c) 2016 by Contributors
 * \file signed_sqrt.cu
 * \brief
 * \author Chen Zhu
*/
#include "./signed_sqrt-inl.h"

namespace mxnet{
namespace op{
template<>
Operator *CreateOp<gpu>(SignedSqrtParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SignedSqrtOp<gpu, DType>(param);
  })
  return op;
}
} // namespace op
} // namespace mxnet