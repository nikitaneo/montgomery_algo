#include <iostream>
#include <string>
#include <uint128_t.h>
#include <random>
#include <assert.h>
//#include <cuda_runtime.h>

uint64_t inline get_bit(const uint64_t* arr, uint64_t n)
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
void montgomery( const T *x, const T *y, unsigned size, const T n, const T k, const T r_1, T *result )
{
	for( unsigned idx = 0; idx < size; idx++ )
	{
		T x_tmp = (x[idx] << k) % n;
		T y_tmp = (y[idx] << k) % n;
		uint64_t xk[2] = {x_tmp.lower(), x_tmp.upper()};

		T A = 0;
		for( unsigned i = 0; i < k; i++ )
		{
			A = A + get_bit(xk, i) * y_tmp;
			A = (A + (A & 1) * n) >> 1;
		}
		
		result[idx] = A > n ? A - n : A;
	}
}

template<typename T>
T inline reduce( const T x, const T r, const T n, const T n_ )
{
	T q = ((x % r) * n_) % r;
	T a = (x + q * n) / r;
	return a > n ? a - n : a;
}

int main()
{
    // r = 2^k
	//uint128_t n(105751);
    uint128_t n(355395216820ULL, 15639199647881268121ULL);
	uint128_t r = 1;
	uint128_t k = 0;
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
	std::cout << "GCD OK. r = " << r << ", r^(-1) = " << r_1 << ", n = " << n << ", n' = " << n_ << std::endl;

    uint128_t *h_result = new uint128_t[1024 * 128];
	uint128_t *h_a      = new uint128_t[1024 * 128];
	uint128_t *h_b      = new uint128_t[1048 * 128];

    //uint128_t *d_result = nullptr, *d_a = nullptr, *d_b = nullptr;
    //cudaMalloc((void **)&d_result, sizeof(uint128_t) * 1024 * 128);
    //cudaMalloc((void **)&d_b, sizeof(uint128_t) * 1024 * 128);
 	//cudaMalloc((void **)&d_a, sizeof(uint128_t) * 1024 * 128);
	
	std::default_random_engine gen;
	std::uniform_int_distribution<uint64_t> dist(1, 4096);
	for( unsigned i = 0; i < 1024 * 128; i++ )
    {
        h_a[i] = dist(gen);
		h_b[i] = dist(gen);
    }

    //cudaMemcpy((void *)d_a, (const void*)h_a, sizeof(uint128_t) * 1024 * 128, cudaMemcpyHostToDevice);
    //cudaMemcpy((void *)d_b, (const void*)h_b, sizeof(uint128_t) * 1024 * 128, cudaMemcpyHostToDevice);

    montgomery( h_a, h_b, 1024 * 128, n, k, r_1, h_result);
	/*cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();    
    cudaMemcpy((void *)h_result, (const void*)d_result, sizeof(uint128_t) * 1024 * 128, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();    
	*/
	for( unsigned i = 0; i < 1024 * 128; i++ )
    {
		if( ((h_a[i] * h_b[i]) << k) % n != h_result[i] )
		{
			std::cout << ((h_a[i] * h_b[i]) << k) % n << " " << h_result[i] << " " << i << std::endl;
			return 1;
		}
    }

	std::cout << "OK" << std::endl;

    delete h_result;
	delete h_a;
	delete h_b;

	//cudaFree(h_result);
	//cudaFree(h_a);
	//cudaFree(h_b);

	return 0;
}
