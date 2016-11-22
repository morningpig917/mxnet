/*!
 * Copyright (c) 2016 by Contributors
 * \file signed_sqrt-inl.h
 * \brief SignedSqrt operator
 * \author Chen Zhu
*/
#ifndef MXNET_OPERATOR_SIGNED_SQRT_INL_H_
#define MXNET_OPERATOR_SIGNED_SQRT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#include <iostream>

namespace mxnet {
namespace op {
// Declare enumeration of input order to make code more intuitive.
// // These enums are only visible within this header
namespace ssqrt {
enum SignedSqrtOpInputs {kData};
enum SignedSqrtOpOutputs {kOut};
}  // ssqrt

struct SignedSqrtParam : public dmlc::Parameter<SignedSqrtParam> {
  float epsilon;
  DMLC_DECLARE_PARAMETER(SignedSqrtParam) {
    DMLC_DECLARE_FIELD(epsilon).set_default(0.06)
    .describe("Epsilon to prevent 1/0 and clip gradient (when large).");
  }
};

/**
 * \brief This is the implementation of ssqrt operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class SignedSqrtOp : public Operator {
 public:
  explicit SignedSqrtOp(SignedSqrtParam p) {
    this->param_ = p;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> data = in_data[ssqrt::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[ssqrt::kOut].FlatTo2D<xpu, DType>(s);
    Assign(out, req[ssqrt::kOut], F<mshadow_op::signed_square_root>(data));
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
    CHECK(in_data.size() == 1 && in_grad.size() == 1);
    CHECK_EQ(req.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> m_out_grad = out_grad[ssqrt::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_out_data = out_data[ssqrt::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> m_in_grad = in_grad[ssqrt::kData].FlatTo2D<xpu, DType>(s);

    //std::cout<<"Dtype(param_.epsilon)= "<<DType(param_.epsilon)<<std::endl;
    //printf("Dtype(param_.epsilon)= %f")

    Assign(m_out_data, req[ssqrt::kData], F<mshadow_op::abs>(m_out_data));
    Assign(m_out_data, req[ssqrt::kData], F<mshadow::op::mul>(m_out_data, DType(2.0)));
    Assign(m_out_data, req[ssqrt::kData], F<mshadow::op::plus>(m_out_data, DType(param_.epsilon)))
    Assign(m_in_grad, req[ssqrt::kData], F<mshadow::op::div>(m_out_grad, m_out_data));
  }
  SignedSqrtParam param_;
};  // class SignedSqrtOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SignedSqrtParam params, int dtype);

#if DMLC_USE_CXX11
class SignedSqrtProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
    const TShape &dshape = in_shape->at(ssqrt::kData);
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
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
    auto ptr = new SignedSqrtProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SignedSqrt";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[ssqrt::kOut], out_data[ssqrt::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[ssqrt::kOut], in_grad[ssqrt::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[ssqrt::kData], out_data[ssqrt::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SignedSqrtParam param_;
};
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SIGNED_SQRT_INL_H_
