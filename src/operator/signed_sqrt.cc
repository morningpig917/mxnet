/*!
 * Copyright (c) 2016 by Contributors
 * \file signed_sqrt.cc
 * \brief signed_sqrt op
 * \author Bing Xu
*/
#include "./signed_sqrt-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SignedSqrtParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SignedSqrtOp<cpu, DType>(param);
  })
  return op;
}

Operator *SignedSqrtProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                      std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SignedSqrtParam);

MXNET_REGISTER_OP_PROPERTY(SignedSqrt, SignedSqrtProp)
.describe(R"(Signed Square Root. 
  Take the signed square root of the input.
  Use epsilon to prevent 1/0 and control the magnitude of gradients.)"
).add_arguments(SignedSqrtParam::__FIELDS__());

}
}