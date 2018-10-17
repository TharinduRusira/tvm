/*!
 *  Copyright (c) 2017 by Contributors
 * \file reduce.cc
 * \brief reduce operator.
 */
<<<<<<< HEAD
=======
// Enforce TOPI to use old behavior that reduces to at least 1d
#define TOPI_REDUCE_ATLEAST1D 1

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include <nnvm/compiler/util.h>
#include <nnvm/top/tensor.h>
<<<<<<< HEAD
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/reduction.h"
=======
#include <numeric>
#include "../op_common.h"
#include "../elemwise_op_common.h"
#include "topi/detail/constant_utils.h"
#include "topi/elemwise.h"
#include "topi/reduction.h"
#include "topi/transform.h"

static_assert(TOPI_REDUCE_ATLEAST1D, "need to use legacy reduce behavior");
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

namespace nnvm {
namespace top {
using namespace tvm;
using namespace nnvm::compiler;

// reduce
DMLC_REGISTER_PARAMETER(ReduceParam);

<<<<<<< HEAD
inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  if (axis.ndim() == 0) {
    if (keepdims) {
      return TShape(ishape.ndim());
    } else {
      return TShape(1);
    }
  }
  CHECK_LT(axis[axis.ndim() - 1], ishape.ndim())
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " Exceeds input dimensions " << ishape;

  TShape in_axis = axis;
  for (auto& i : in_axis) {
    i = i < 0 ? i + ishape.ndim(): i;
    CHECK_GE(i, 0) << "axis out of bounds in reduce operator";
    CHECK_LT(i, ishape.ndim()) << "axis out of bounds in reduce operator";
  }
  std::sort(in_axis.begin(), in_axis.end());

  if (keepdims) {
    TShape oshape(ishape);
    if (exclude) {
      for (dim_t i = 0, j = 0; i < ishape.ndim(); ++i) {
        if (j < in_axis.ndim() && i == in_axis[j]) {
          ++j;
          continue;
        }
        oshape[i] = 1;
      }
      return oshape;
    }

    for (dim_t i = 0; i < in_axis.ndim(); ++i) {
      oshape[in_axis[i]] = 1;
=======
inline TShape GetReduceAxes(const uint32_t indim,
                            const TShape& axis,
                            bool exclude) {
  if (axis.ndim() == 0) {
    TShape r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  CHECK_LT(axis[axis.ndim() - 1], indim)
    << "Reduction axis " << axis[axis.ndim() - 1]
    << " exceeds input dimensions " << indim;

  TShape in_axis = axis;
  for (auto& i : in_axis) {
    i = i < 0 ? i + indim : i;
    CHECK_GE(i, 0) << "axis out of bounds in reduce operator";
    CHECK_LT(i, indim) << "axis out of bounds in reduce operator";
  }
  std::sort(in_axis.begin(), in_axis.end());
  if (!exclude) return in_axis;
  TShape r_axis(indim - in_axis.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axis.ndim() && i == in_axis[j]) {
        ++j;
        continue;
    }
    r_axis[k++] = i;
  }
  return r_axis;
}

inline TShape ReduceShapeImpl(const TShape& ishape,
                              const TShape& axis,
                              bool keepdims,
                              bool exclude) {
  uint32_t indim = ishape.ndim();
  TShape r_axes = GetReduceAxes(indim, axis, exclude);
  if (!r_axes.ndim()) return ishape;
  if (r_axes.ndim() == indim)
    return TShape(keepdims ? indim : 1);

  CHECK(r_axes.ndim() < indim);
  if (keepdims) {
    TShape oshape(ishape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (j >= r_axes.ndim() || i != r_axes[j]) continue;
      oshape[i] = 1;
      ++j;
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    }
    return oshape;
  }

<<<<<<< HEAD
  if (exclude) {
    TShape oshape = TShape(in_axis.ndim());
    for (dim_t i = 0; i < in_axis.ndim(); ++i) {
      oshape[i] = ishape[in_axis[i]];
    }
    return oshape;
  }
  TShape oshape = TShape(std::max<dim_t>(1, ishape.ndim() - in_axis.ndim()));
  for (dim_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
    if (j < in_axis.ndim() && i == in_axis[j]) {
=======
  TShape oshape(indim - r_axes.ndim());
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < r_axes.ndim() && i == r_axes[j]) {
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
      ++j;
      continue;
    }
    oshape[k++] = ishape[i];
  }
  return oshape;
}

inline bool ReduceShape(const nnvm::NodeAttrs& attrs,
                        std::vector<TShape>* in_attrs,
                        std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 0) return false;
  const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
<<<<<<< HEAD
  NNVM_ASSIGN_INPUT_SHAPE(
=======
  NNVM_ASSIGN_OUTPUT_SHAPE(
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
      attrs, *out_attrs, 0,
      ReduceShapeImpl((*in_attrs)[0], param.axis,
                      param.keepdims, param.exclude));
  return true;
}

<<<<<<< HEAD
=======
inline bool CollapseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape>* in_attrs,
                          std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  if ((*in_attrs)[0].ndim() == 1) return false;
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, (*in_attrs)[1]);
  return true;
}

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
template<typename PType>
inline void AxesParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  param.Init(attrs->dict);
  std::sort(&param.axis[0], &param.axis[param.axis.ndim()]);
  attrs->parsed = std::move(param);
}

<<<<<<< HEAD
#define NNVM_REGISTER_REDUCE_OP(op)                                     \
  NNVM_REGISTER_OP(op)                                                  \
  .add_argument("data", "Tensor", "The input")                          \
  .add_arguments(ReduceParam::__FIELDS__())                             \
  .set_attr_parser(AxesParamParser<ReduceParam>)                        \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
=======
#define NNVM_REGISTER_BASE_REDUCE_OP(op)                                 \
  NNVM_REGISTER_OP(op)                                                   \
  .add_arguments(ReduceParam::__FIELDS__())                              \
  .set_attr_parser(AxesParamParser<ReduceParam>)                         \
  .set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<ReduceParam>) \
  .set_num_outputs(1)

#define NNVM_REGISTER_REDUCE_OP(op)                                     \
  NNVM_REGISTER_BASE_REDUCE_OP(op)                                      \
  .add_argument("data", "Tensor", "The input")                          \
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
  .set_attr<FInferShape>("FInferShape", ReduceShape)                    \
  .set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)               \
  .set_attr<FCorrectLayout>("FCorrectLayout",                           \
    ElemwiseFixedLayoutUnknownOut<1, 1>)                                \
<<<<<<< HEAD
  .set_num_inputs(1)                                                    \
  .set_num_outputs(1)
=======
  .set_num_inputs(1)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

NNVM_REGISTER_REDUCE_OP(sum)
.describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
<<<<<<< HEAD
    Array<Expr> axis;
    if (param.exclude) {
      std::set<dim_t> exclude_axis;
      for (dim_t i = 0; i < param.axis.ndim(); ++i) {
        exclude_axis.insert(param.axis[i]);
      }
      for (dim_t i = 0; i < static_cast<int>(inputs[0].ndim()); ++i) {
        if (exclude_axis.count(i) == 0) {
          axis.push_back(make_const(Int(32), i));
        }
      }
    } else {
      axis = ShapeToArray(param.axis);
    }
=======
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    if (!r_axes.ndim()) return Array<Tensor> { topi::identity(inputs[0]) };
    auto axis = ShapeToArray(r_axes);
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    return Array<Tensor>{
      topi::sum(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
<<<<<<< HEAD
    std::ostringstream axis; axis << param.axis;
=======
    bool exclude = param.exclude;
    TShape p_axis = param.axis;
    if (!param.exclude && param.axis.ndim() == 0) {
      exclude = true;
      p_axis = TShape();
    }
    std::ostringstream axis; axis << p_axis;
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    return std::vector<NodeEntry>{
      MakeNode("expand_like", n->attrs.name + "_grad",
               {ograds[0], n->inputs[0]},
               {{"axis", axis.str()},
<<<<<<< HEAD
                {"exclude", std::to_string(param.exclude)}})
=======
                {"exclude", std::to_string(exclude)}})
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
  };
});

NNVM_REGISTER_REDUCE_OP(max)
.describe(R"code(Computes the max of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
<<<<<<< HEAD
    auto axis = ShapeToArray(param.axis);
=======
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    return Array<Tensor>{
      topi::max(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                             {ograds[0], n->inputs[0]},
                             {{"axis", axis.str()},
<<<<<<< HEAD
                              {"keepdims", std::to_string(param.keepdims)},
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                              {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_max_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});

NNVM_REGISTER_REDUCE_OP(min)
.describe(R"code(Computes the min of array elements over given axes.

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
<<<<<<< HEAD
    auto axis = ShapeToArray(param.axis);
=======
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    return Array<Tensor>{
      topi::min(inputs[0], axis, param.keepdims) };
})
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds){
    const ReduceParam& param = nnvm::get<ReduceParam>(n->attrs.parsed);
    std::ostringstream axis; axis << param.axis;
    NodeEntry sub0 = MakeNode("expand_like", n->attrs.name + "_grad_sub0",
                              {ograds[0], n->inputs[0]},
                              {{"axis", axis.str()},
<<<<<<< HEAD
                               {"keepdims", std::to_string(param.keepdims)},
=======
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
                               {"exclude", std::to_string(param.exclude)}});
    NodeEntry sub1 = MakeNode("_min_mask", n->attrs.name + "_grad_sub1",
                              {ograds[0]},
                              {{"axis", axis.str()},
                               {"exclude", std::to_string(param.exclude)}});
    return std::vector<NodeEntry>{
      MakeNode("elemwise_mul", n->attrs.name + "_grad", {sub0, sub1})
    };
});

<<<<<<< HEAD
=======
NNVM_REGISTER_BASE_REDUCE_OP(collapse_sum)
.add_argument("data", "Tensor", "The input")
.add_argument("as", "Tensor", "The reference")
.set_attr<FInferShape>("FInferShape", CollapseShape)
.set_attr<FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<2, 1>)
.set_num_inputs(2)
.describe(R"code(Reduces lhs to the shape of rhs via sum)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    return Array<Tensor>{ topi::collapse_sum(inputs[0], inputs[1]->shape) };
});

template<int Type>
inline bool InferFixedType(const NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
  // Static type inference for argmax operation. Argmax return indices which
  // should have Int32 type as shapes do.
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  NNVM_ASSIGN_OUTPUT_TYPE(attrs, *out_attrs, 0, static_cast<int>(Type));
  return true;
}

NNVM_REGISTER_BASE_REDUCE_OP(argmax)
.describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "The input")
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FInferType>("FInferType", InferFixedType<kInt32>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::argmax(inputs[0], axis, param.keepdims) };
});

NNVM_REGISTER_BASE_REDUCE_OP(argmin)
.describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" NNVM_ADD_FILELINE)
.add_argument("data", "Tensor", "The input")
.set_attr<FInferShape>("FInferShape", ReduceShape)
.set_attr<FInferType>("FInferType", InferFixedType<kInt32>)
.set_attr<FCorrectLayout>("FCorrectLayout", ElemwiseFixedLayoutUnknownOut<1, 1>)
.set_num_inputs(1)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::argmin(inputs[0], axis, param.keepdims) };
});

NNVM_REGISTER_REDUCE_OP(mean)
  .describe(R"code(Computes the mean of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data)
  [3.22]

  mean(data, axis=[1,2])
  [ 2.  3.16666667  4.5]

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    if (!r_axes.ndim()) return Array<Tensor> { topi::identity(inputs[0]) };
    auto axis = ShapeToArray(r_axes);

    Expr count = make_const(inputs[0]->dtype, 1);
    for (auto& i : r_axes) {
      count *= inputs[0]->shape[i];
    }

    return Array<Tensor>{
      topi::divide(topi::sum(inputs[0], axis, param.keepdims), count) };
});

NNVM_REGISTER_REDUCE_OP(prod)
  .describe(R"code(Computes the products of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data, axis=1)
  [35562240]

  mean(data, axis=[1,2])
  [ 36  480  2058]

)code" NNVM_ADD_FILELINE)
.set_attr<FTVMCompute>(
  "FTVMCompute", [](const NodeAttrs& attrs,
                    const Array<Tensor>& inputs,
                    const Array<Tensor>& out_info) {
    const ReduceParam& param = nnvm::get<ReduceParam>(attrs.parsed);
    TShape r_axes = GetReduceAxes(inputs[0]->shape.size(),
                                  param.axis, param.exclude);
    if (!r_axes.ndim()) return Array<Tensor> { topi::identity(inputs[0]) };
    auto axis = ShapeToArray(r_axes);
    return Array<Tensor>{
      topi::prod(inputs[0], axis, param.keepdims) };
});

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

}  // namespace top
}  // namespace nnvm
