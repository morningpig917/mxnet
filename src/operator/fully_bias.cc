#include "./fully_bias-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(FullyBiasParam param, int dtype) {
  Operator *op = NULL;
  switch (dtype) {
  case mshadow::kFloat32:
    op = new FullyBiasOp<cpu, float>(param);
    break;
  case mshadow::kFloat64:
    op = new FullyBiasOp<cpu, double>(param);
    break;
  case mshadow::kFloat16:
    LOG(FATAL) << "float16 fully bias layer is currently"
                  "only supported by CuDNN version.";
    break;
  default:
    LOG(FATAL) << "Unsupported type " << dtype;
  }
  return op;
}
Operator *FullyBiasProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(FullyBiasParam);

MXNET_REGISTER_OP_PROPERTY(FullyBias, FullyBiasProp)
.describe(R"(Add a bias to the input from fully connected layer, in
            equavalence to adding an additional fc layer to embed the answers.)")
.add_argument("data", "Symbol", "Input data to the FullyBiasOp.")
.add_argument("bias_weight", "Symbol", "bias matrix.")
.add_arguments(FullyBiasParam::__FIELDS__());
}
}