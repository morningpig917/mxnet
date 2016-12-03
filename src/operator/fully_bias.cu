#include "./fully_bias-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(FullyBiasParam param, int dtype)
{
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new FullyBiasOp<gpu, DType>(param);
  })
  return op;
}
} // namespace op
} // namespace mxnet