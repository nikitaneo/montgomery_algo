#include <iostream>
#include <string>
#include <uint128_t.h>
#include <random>
#include <assert.h>
#include <cuda_runtime.h>
#include <chrono>

__host__ __device__ uint64_t inline get_bit(const uint64_t* arr, uint64_t n)
{
	return (arr[n / 64] >> (n % 64)) & 1;
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
__global__ void montgomery_gpu( const T *x, const T *y, unsigned size, const T n, const T r, unsigned k, const T r_1, const T n_, T *result )
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

	T x_tmp = (x[idx] << k) % n;
	T y_tmp = (y[idx] << k) % n;
	uint64_t xk[2] = {std::move(x_tmp.lower()), std::move(x_tmp.upper())};

	T A = 0;
	for( unsigned i = 0; i < k; i++ )
	{
		A = A + y_tmp * get_bit(xk, i);
		A = (A + (A & 1) * n) >> 1;
	}
		
	result[idx] = reduce(A > n ? A - n : A, r, n, n_);
}


template<typename T> 
void montgomery_cpu( const T *x, const T *y, unsigned size, const T n, const T r, unsigned k, const T r_1, const T n_, T *result )
{
	for( unsigned idx = 0; idx < size; idx++ )
	{
		T x_tmp = (x[idx] << k) % n;
		T y_tmp = (y[idx] << k) % n;
		uint64_t xk[2] = {std::move(x_tmp.lower()), std::move(x_tmp.upper())};

		T A = 0;
		for( unsigned i = 0; i < k; i++ )
		{
			A = A + get_bit(xk, i) * y_tmp;
			A = (A + (A & 1) * n) >> 1;
		}
		
		result[idx] = reduce(A > n ? A - n : A, r, n, n_);
	}
}

template<typename T> 
__global__ void naive_gpu( const T *x, const T *y, unsigned size, const T n, T *result )
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	result[idx] = (x[idx] * y[idx]) % n;
}

template<typename T> 
void naive( const T *x, const T *y, unsigned size, const T n, T *result )
{
	for( unsigned idx = 0; idx < size; idx++ )
	{
		result[idx] = (x[idx] * y[idx]) % n;
	}
}

template<typename T>
__host__ __device__ inline T reduce( const T &x, const T &r, const T &n, const T &n_ )
{
	T a = (x + (((x % r) * n_) % r) * n) / r;
	return a > n ? a - n : a;
}

int main()
{
    // r = 2^k
	//uint128_t n(105751);
    uint128_t n(14797879391564958223ULL);
	uint128_t r = 1;
	unsigned k = 0;

	while( r < n )
	{
		r <<= 1;
		k++;
	}

	assert( uint128_t(1) << k == r );

    uint128_t r_1 = 0;
    uint128_t n_ = 0;

	xbinGCD( r / 2, n, r_1, n_ );

	assert( r * r_1 - n * n_ == 1 );

	uint128_t *device_result_mont = new uint128_t[1024 * 128];
	uint128_t *device_result_naive = new uint128_t[1024 * 128];
	uint128_t *h_result = new uint128_t[1024 * 128];
	uint128_t *naive_result = new uint128_t[1024 * 128];
	uint128_t *h_a = new uint128_t[1024 * 128];
	uint128_t *h_b = new uint128_t[1048 * 128];

    uint128_t *d_result = nullptr, *d_naive_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaMalloc((void **)&d_result, sizeof(uint128_t) * 1024 * 128);
	cudaMalloc((void **)&d_naive_result, sizeof(uint128_t) * 1024 * 128);
    cudaMalloc((void **)&d_b, sizeof(uint128_t) * 1024 * 128);
 	cudaMalloc((void **)&d_a, sizeof(uint128_t) * 1024 * 128);
	
	std::default_random_engine gen;
	std::uniform_int_distribution<uint64_t> dist(1, uint64_t(-1));
	for( unsigned i = 0; i < 1024 * 128; i++ )
    {
        h_a[i] = dist(gen);
		h_b[i] = dist(gen);
    }

    cudaMemcpy((void *)d_a, (const void*)h_a, sizeof(uint128_t) * 1024 * 128, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (const void*)h_b, sizeof(uint128_t) * 1024 * 128, cudaMemcpyHostToDevice);

	/*
		Launch Montgomery multiplication algorithm with CUDA
	*/
	auto gpu_time_begin = std::chrono::steady_clock::now();
    montgomery_gpu<<<128, 1024>>>( d_a, d_b, 1024 * 128, n, r, k, r_1, n_, d_result );
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
	auto gpu_time_end = std::chrono::steady_clock::now();
	auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_time_end - gpu_time_begin).count();

    cudaMemcpy((void *)device_result_mont, (const void*)d_result, sizeof(uint128_t) * 1024 * 128, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	/*
		Launch naive multiplication algorithm with CUDA
	*/
	auto gpu_naive_time_begin = std::chrono::steady_clock::now();
    naive_gpu<<<128, 1024>>>( d_a, d_b, 1024 * 128, n, d_naive_result );
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
	auto gpu_naive_time_end = std::chrono::steady_clock::now();
	auto gpu_naive_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_naive_time_end - gpu_naive_time_begin).count();

    cudaMemcpy((void *)device_result_naive, (const void*)d_naive_result, sizeof(uint128_t) * 1024 * 128, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// Launch Montgomery multiplication algorithm on CPU
	auto cpu_mont_time_begin = std::chrono::steady_clock::now();
	montgomery_cpu( h_a, h_b, 1024 * 128, n, r, k, r_1, n_, h_result );
	auto cpu_mont_time_end = std::chrono::steady_clock::now();
	auto cpu_mont_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_mont_time_end - cpu_mont_time_begin).count();

	// Launch naive multiplication algorithm on CPU
	auto cpu_naive_time_begin = std::chrono::steady_clock::now();
	naive( h_a, h_b, 1024 * 128, n, naive_result );
	auto cpu_naive_time_end = std::chrono::steady_clock::now();
	auto cpu_naive_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_naive_time_end - cpu_naive_time_begin).count();

	for( unsigned i = 0; i < 1024 * 128; i++ )
    {
		assert( h_result[i] == device_result_mont[i] && h_result[i] == naive_result[i] && device_result_naive[i] == naive_result[i] );
    }

	std::cout << "128 * 1024 multiplications" << std::endl;
	std::cout << "GPU Montgomery time: " << gpu_time << std::endl;
	std::cout << "GPU naive time: " << gpu_naive_time << std::endl;
	std::cout << "CPU Montgomery time: " << cpu_mont_time << std::endl;
	std::cout << "CPU naive time: " << cpu_naive_time << std::endl;

	delete[] device_result_mont;
	delete[] device_result_naive;
	delete[] h_result;
	delete[] naive_result;
	delete[] h_a;
	delete[] h_b;

	cudaFree(d_result);
	cudaFree(d_naive_result);
	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}
