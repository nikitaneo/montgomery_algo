#include <iostream>
#include <string>
#include <uint128_t.h>
#include <random>
#include <assert.h>
#include <cuda_runtime.h>

// Solve a*x + b*y = d 
template<typename T>
T gcd( const T &a, const T &b, T &x, T &y )
{
	if(a == 0)
    {
		x = 0;
        y = 1;
		return b;
	}
	T x1, y1;
	T d = gcd( b%a, a, x1, y1 );
	x = y1 - (b / a) * x1;
	y = x1;
	return d;
}

template<typename T> 
__global__ void fastMul( const T *a, const T *b, unsigned size, const T n, const T n_, int k, const T r_1, T *result )
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx > size )
		return;

	// to Montgomery representation
	T lhs = (a[idx] << k) % n;
	T rhs = (b[idx] << k) % n;

    T t = lhs * rhs;
    T u = (t + (t * n_ % (1 << k)) * n) * r_1;
    u = u > n ? u - n : u;
	
	// back to common repr
	result[idx] = u * r_1 % n; 
}

int main()
{
    // r = 2^k
    int k = 11;
    uint128_t n(65558846097UL, 14695357921182649241UL);

	std::cout << n << std::endl;

    uint128_t r_1 = 0;
    uint128_t n_ = 0;
    
    gcd( uint128_t(1) << k, n, r_1, n_ );
    r_1 = -r_1;
   
    uint128_t *h_result = new uint128_t[1024 * 128];
	uint128_t *h_a      = new uint128_t[1024 * 128];
	uint128_t *h_b      = new uint128_t[1048 * 128];

    uint128_t *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaMalloc((void **)&d_result, sizeof(uint128_t) * 1024 * 128);
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

    fastMul<<<128, 1024>>>( d_a, d_b, 1024 * 128, n, n_, k, r_1, d_result);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();    
    cudaMemcpy((void *)h_result, (const void*)d_result, sizeof(uint128_t) * 1024 * 128, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();    

	for( unsigned i = 0; i < 1024 * 128; i++ )
    {
		if( (h_a[i] * h_b[i]) % n != h_result[i] )
		{
			std::cout << (h_a[i] * h_b[i]) % n << " " << h_result[i] << " " << i << std::endl;
			return 0;
		}
    }

	std::cout << "OK" << std::endl;

    delete h_result;
	delete h_a;
	delete h_b;

	cudaFree(h_result);
	cudaFree(h_a);
	cudaFree(h_b);

	return 1;
}
