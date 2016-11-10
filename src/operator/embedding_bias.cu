/*!
 * Copyright (c) 2015 by Contributors
 * \file embedding_bias.cu
 * \brief
 * \author Bing Xu
*/

#include "./embedding_bias-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(EmbeddingBiasParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new EmbeddingBiasOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet

