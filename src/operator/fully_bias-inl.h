#ifndef MXNET_OPERATOR_FULLY_BIAS_INL_H_
#define MXNET_OPERATOR_FULLY_BIAS_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet{
namespace op{
namespace fullb{
  enum FullyBiasOpInputs {kData, kBias};
  enum FullyConnectedOpOutputs {kOut};
}

struct FullyBiasParam : public dmlc::Parameter<FullyBiasParam> {
  int num_output;
  DMLC_DECLARE_PARAMETER(FullyBiasParam) {
    DMLC_DECLARE_FIELD(num_output).set_lower_bound(1)
    .describe("Number of class labels to pad to the last dimension.");
  }
};

template<typename xpu, typename DType>
class FullyBiasOp : public Operator {
 public:
  explicit FullyBiasOp(FullyBiasParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    if (req[fullb::kOut] == kNullOp) return;
    CHECK_EQ(req[fullb::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif  // __CUDACC__
    const TShape& ishape = in_data[fullb::kData].shape_;
    const TShape& oshape = out_data[fullb::kOut].shape_;
    Tensor<xpu, 3, DType> data = in_data[fullb::kData].get_with_shape<xpu, 3, DType>(
        Shape3(ishape[0], ishape.ProdShape(1, ishape.ndim()), 1), s);
    Tensor<xpu, 3, DType> out = out_data[fullb::kOut].get_with_shape<xpu, 3, DType>(
        Shape3(oshape[0], oshape[1], oshape[2]), s);
    Tensor<xpu, 3, DType> bias = in_data[fullb::kBias].get_with_shape<xpu, 3, DType>(
        Shape3(1, ishape.ProdShape(1, ishape.ndim()), param_.num_output));
    out = F<mshadow::op::plus>(broadcast_to(data, oshape), broadcast_to(bias, oshape));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK(in_data.size() == 2 && in_grad.size() == 2);
    CHECK_EQ(req.size(), 2);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data[fullb::kData].shape_;
    const TShape& oshape = out_grad[fullb::kOut].shape_;

    // grad with another shape to compute gbias
    Tensor<xpu, 1, DType> gbias = in_grad[fullb::kBias].get_with_shape<xpu, 1, DType>(
        Shape1(oshape.ProdShape(1, oshape.ndim())), s);
    Tensor<xpu, 2, DType> grad = out_grad[fullb::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
    Assign(gbias, req[fullb::kBias], sum_rows(grad));

    grad = out_grad[fullb::kOut].get_with_shape<xpu, 2, DType>(
        Shape2(oshape[0]*oshape[1], oshape.ProdShape(2, oshape.ndim())), s);
    Tensor<xpu, 1, DType> gdata = in_grad[fullb::kData].get_with_shape<xpu, 1, DType>(
        Shape1(ishape.ProdShape(0, ishape.ndim())), s);
    Assign(gdata, req[fullb::kData], sumall_except_dim<0>(grad));
  }
 private:
  FullyBiasParam param_;
}; // class FullyBiasOp
 
template<typename xpu>
Operator* CreateOp(FullyBiasParam param, int dtype);

#if DMLC_USE_CXX11
class FullyBiasProp : public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "bias_weight"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, bias_weight]";
    const TShape &dshape = (*in_shape)[fullb::kData];
    // require data to be known
    if (dshape.ndim() == 0) return false;

    index_t num_input = dshape.ProdShape(1, dshape.ndim());
    SHAPE_ASSIGN_CHECK(*in_shape, fullb::kBias, Shape3(1, num_input, param_.num_output));
    out_shape->clear();
    out_shape->push_back(Shape3(dshape[0], dshape[1], param_.num_output));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    FullyBiasProp* fb_sym = new FullyBiasProp();
    fb_sym->param_ = this->param_;
    return fb_sym;
  }

  std::string TypeString() const override {
    return "FullyBias";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[fullb::kOut], in_data[fullb::kData], in_data[fullb::kBias]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{in_data[fullb::kData], in_grad[fullb::kData]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
private:
  FullyBiasParam param_;
}; // class FullyBiasSymbol
#endif
} // namespace op
} // namespace mxnet
#endif // fully bias