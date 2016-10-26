/*!
 * Copyright (c) 2015 by Contributors
 * \file count_sketch.cu
 * \brief count_sketch op
 * \author Chen Zhu
*/
#include <algorithm>
#include "./count_sketch-inl.h"
#include <mshadow/tensor.h>

#define WARPS_PER_BLOCK 1
#define THREADS_PER_BLOCK 512

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
namespace mshadow {
namespace cuda {
// wrappers to deal with atomic add 
// supporting only single precision
__device__ void atomic_add(float* dst, float val) {
	atomicAdd(dst, val);
}

// for double precision
__device__ void atomic_add(double* address, double val) {
  // code example in the official document at:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  //      #atomic-functions

  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  // NOLINT_NEXT_LINE(runtime/int)
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
}

template <typename DType>
__global__ void sketch_forward_kernel(const int nthreads, DType *out, const DType *h,
					const DType *s, const DType *in, const int n_smaples, 
					const int in_dim, const int out_dim) {
	// input: n_smaples * in_dim 
	// output: n_smaples * out_dim
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	// nthreads is the maximum of thread indices, should be equal to in_dim
	// index is point index
	const int i_indim = index % in_dim;
	const int i_sample = index / in_dim;

	// get the target location in the output
	const int target = i_sample*out_dim + h[i_indim];
	atomic_add(out + target, s[i_indim] * in[index]);
	//out[target] = s[i_indim] * in[index];
}

template <typename DType>
__global__ void sketch_backward_kernel(const int nthreads, DType *in_grad, const DType *h,
					const DType *s, const DType *out_grad, const int n_smaples, 
					const int in_dim, const int out_dim) {
	// input: n_smaples * in_dim 
	// output: n_smaples * out_dim
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int i_indim = index % in_dim;
	const int i_sample = index / in_dim;
	const int i_outdim = i_sample*out_dim + h[i_indim];
	in_grad[index] = out_grad[i_outdim] * s[i_indim];
}

} // namespace cuda

// CountSketch Forward
template <typename DType>
inline void CountSketchForward(Tensor<gpu, 2, DType> &out,
															 const Tensor<gpu, 2, DType> &in,
															 const Tensor<gpu, 1, DType> &h,
															 const Tensor<gpu, 1, DType> &s,
															 const int n_samples,
															 const int processing_batch_size, 
															 const int in_dim,
															 const int out_dim) {
	DType *out_ptr = out.dptr_;
	const DType *in_ptr = in.dptr_;
	const DType *h_ptr = h.dptr_;
	const DType *s_ptr = s.dptr_;
	for ( int bstart = 0; bstart < n_samples; bstart += processing_batch_size ){
		const int batchlen = min(processing_batch_size, n_samples - bstart );
		const int nthreads = batchlen * in_dim;
		const int threads_per_block = min(THREADS_PER_BLOCK, nthreads);// to make number of threads the same as input 
		int nblocks = (nthreads - threads_per_block + 1) / threads_per_block + 1;

		cuda::sketch_forward_kernel<DType><<<nblocks, threads_per_block>>>(
									nthreads, out_ptr+bstart*out_dim, h_ptr,
									s_ptr, in_ptr+bstart*in_dim, batchlen, 
									in_dim, out_dim);

	}
}

template<typename DType>
inline void CountSketchBackward(Tensor<gpu, 2, DType> &in_grad,
																const Tensor<gpu, 2, DType> &out_grad,
																const Tensor<gpu, 1, DType> &h,
																const Tensor<gpu, 1, DType> &s, 
																const int n_samples,
																const int processing_batch_size,
																const int in_dim,
																const int out_dim	) {
	DType *in_grad_ptr = in_grad.dptr_;
	const DType *out_grad_ptr = out_grad.dptr_;
	const DType *h_ptr = h.dptr_;
	const DType *s_ptr = s.dptr_;
	for ( int bstart = 0; bstart < n_samples; bstart += processing_batch_size) {
		const int batchlen = min(processing_batch_size, n_samples - bstart);

		const int nthreads = batchlen * in_dim;
		const int threads_per_block = min(THREADS_PER_BLOCK, nthreads);// to make number of threads the same as input 
		int nblocks = (nthreads - threads_per_block + 1) / threads_per_block + 1;
		
		cuda::sketch_backward_kernel<DType><<<nblocks, threads_per_block>>>(
									nthreads, in_grad_ptr+bstart*in_dim, h_ptr,
					        s_ptr, out_grad_ptr+bstart*out_dim, batchlen, 
					        in_dim, out_dim);
	}
}
}
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(CountSketchParam param, int dtype) {
	Operator *op = NULL;
	switch (dtype) {
		case mshadow::kFloat32:
			op = new CountSketchOp<gpu, float>(param);
			break;
		case mshadow::kFloat64:
			op = new CountSketchOp<gpu, double>(param);
			break;
		case mshadow::kFloat16:
			LOG(FATAL) << "float16 count sketch layer is currently"
                  "not supported.";
      break;
    default:
    	LOG(FATAL) << "Unsupported type " << dtype;
	}
	return op;
}
}  // namespace op
}  // namespace mxnet