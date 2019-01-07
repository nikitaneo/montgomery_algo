#ifndef __FASTMUL_H
#define __FASTMUL_H

#include <cuda_runtime.h>

namespace
{

template<typename T>
__host__ __device__ inline T reduce( const T &x, const T &r, unsigned k, const T &n, const T &n_ )
{
	T a = (x + (((x % r) * n_) % r) * n) >> k;
	return a > n ? a - n : a;
}

// Solve a*x - b*y = d 
template<typename T>
void xbinGCD(T a, T b, T &pu, T &pv)
{
	T alpha = a, beta = b, u = 1, v = 0;

	while (a > 0)
	{
		a = a >> 1;
		if ((u & 1) == 0)
		{
			u = u >> 1; v = v >> 1;
		}
		else
		{
			u = ((u ^ beta) >> 1) + (u & beta);
			v = (v >> 1) + alpha;
		}
	}
	pu = u;
	pv = v;
}

template<typename T>
__global__ void montgomery_gpu( const T *x,
								const T *y,
								unsigned size,
								const T n,
								const T r,
								unsigned k,
								const T n_,
								T *result )
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

	T x_tmp = (x[idx] << k) % n;
	T y_tmp = (y[idx] << k) % n;

	T A = 0;
	for( unsigned i = 0; i < k; i++ )
	{
		A = A + y_tmp * ((x_tmp >> i) & 1);
		A = (A + (A & 1) * n) >> 1;
	}

    result[idx] = reduce(A, r, k, n, n_);
}


template<typename T> 
void montgomery_cpu( 	const T *x,
						const T *y,
						unsigned size,
						const T n,
						const T r,
						unsigned k,
						const T n_,
						T *result )
{
	for( unsigned idx = 0; idx < size; idx++ )
	{
		T x_tmp = (x[idx] << k) % n;
		T y_tmp = (y[idx] << k) % n;

		T A = 0;
		for( unsigned i = 0; i < k; i++ )
		{
			A = A + ((x_tmp >> i) & 1) * y_tmp;
			A = (A + (A & 1) * n) >> 1;
		}
		
		result[idx] = reduce(A, r, k, n, n_);
	}
}

}

template<typename T> 
__global__ void naive_gpu( const T *x, const T *y, unsigned size, const T n, T *result )
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	result[idx] = (x[idx] * y[idx]) % n;
}

template<typename T> 
void naive_cpu( const T *x, const T *y, unsigned size, const T n, T *result )
{
	for( unsigned idx = 0; idx < size; idx++ )
	{
		result[idx] = (x[idx] * y[idx]) % n;
	}
}

/*
    Calculate (result[i] = a[i]*b[i] mod n) for each element in input arrays using Montgomery multiplication algorithm. 
    T *result should be already allocated.
*/
template<typename T> 
void modmul_cpu( const T *a, const T *b, unsigned size, const T n, T *result )
{
	T r = 1;
	unsigned k = 0;

	while( r < n )
	{
		r <<= 1;
		k++;
	}

    T r_1 = 0;
    T n_ = 0;

	xbinGCD( r >> 1, n, r_1, n_ );
	
    /*
		Launch Montgomery multiplication algorithm
	*/
    montgomery_cpu( a, b, size, n, r, k, n_, result );
}

template<typename T> 
void modmul_gpu( const T *a, const T *b, unsigned size, const T n, T *result )
{
	T r = 1;
	unsigned k = 0;

	while( r < n )
	{
		r <<= 1;
		k++;
	}

    T r_1 = 0;
    T n_ = 0;

	xbinGCD( r >> 1, n, r_1, n_ );

    T *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaMalloc((void **)&d_result, sizeof(T) * size);
    cudaMalloc((void **)&d_b, sizeof(T) * size);
 	cudaMalloc((void **)&d_a, sizeof(T) * size);

    cudaMemcpy((void *)d_a, (const void*)a, sizeof(T) * size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (const void*)b, sizeof(T) * size, cudaMemcpyHostToDevice);

	dim3 block(1024);
	dim3 grid(size % block.x == 0 ? size / block.x : size / block.x + 1);

    /*
		Launch Montgomery multiplication algorithm with CUDA
	*/
    montgomery_gpu<<<grid, block>>>( d_a, d_b, size, n, r, k, n_, d_result );
	cudaError_t error = cudaGetLastError();
#ifdef DEBUG
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
#endif
	cudaDeviceSynchronize();

    cudaMemcpy((void *)result, (const void*)d_result, sizeof(T) * size, cudaMemcpyDeviceToHost);

	cudaFree(d_result);
	cudaFree(d_a);
	cudaFree(d_b);
}

#endif