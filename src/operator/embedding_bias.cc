/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding.cc
 * \brief
 * \author Bing Xu
*/

#include "./embedding_bias-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(EmbeddingBiasParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new EmbeddingBiasOp<cpu, DType>(param);
  });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *EmbeddingBiasProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(EmbeddingBiasParam);

MXNET_REGISTER_OP_PROPERTY(EmbeddingBias, EmbeddingBiasProp)
.describe("Get biased embedding for one-hot input. A n-dimensional input tensor will "
"be trainsformed into a (n+1)-dimensional tensor, where a new dimension is "
"added for the embedding results.")
.add_argument("data", "Symbol", "Input data to the EmbeddingBiasOp.")
.add_argument("weight", "Symbol", "Enbedding weight matrix.")
.add_argument("bias", "Symbol", "Embedding bias.")
.add_arguments(EmbeddingBiasParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
