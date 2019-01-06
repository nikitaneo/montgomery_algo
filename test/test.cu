#include "fastmul.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <uint128_t.h>
#include <vector>
#include <iostream>
#include <chrono>

TEST(FastMultCPU, Simple)
{
    unsigned size = 128 * 1024;

    std::vector<uint128_t> a( size );
    std::vector<uint128_t> b( size );
    std::vector<uint128_t> c( size );
    std::vector<uint128_t> naive_c( size );
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 gen( seed );
	std::uniform_int_distribution<uint64_t> dist( 1, uint64_t(-1) );
	for( unsigned i = 0; i < size; i++ )
    {
        a[i] = dist( gen );
		b[i] = dist( gen );
    }
    uint128_t mod(14797879391564958223ULL);

    auto begin = std::chrono::steady_clock::now();
    modmul_cpu( a.data(), b.data(), a.size(), mod, c.data() );
    auto end = std::chrono::steady_clock::now();
    auto el_time = std::chrono::duration_cast<std::chrono::microseconds>( end - begin ).count();

    std::cerr << "Montgomery multiplication (CPU) on " << size << " elements has finished with elapsed time " << el_time << " ms." << std::endl;

    for( unsigned i = 0; i < naive_c.size(); i++ )
    {
        naive_c[i] = a[i] * b[i] % mod;
    }

    for( unsigned i = 0; i < c.size(); i++ )
    {
        ASSERT_EQ(c[i], naive_c[i]);
    }
}

TEST(FastMultGPU, Simple)
{
    unsigned size = 128 * 1024;

    std::vector<uint128_t> a( size );
    std::vector<uint128_t> b( size );
    std::vector<uint128_t> c( size );
    std::vector<uint128_t> naive_c( size );
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 gen( seed );
	std::uniform_int_distribution<uint64_t> dist( 1, uint64_t(-1) );
	for( unsigned i = 0; i < size; i++ )
    {
        a[i] = dist( gen );
		b[i] = dist( gen );
    }
    uint128_t mod = 5;

    auto begin = std::chrono::steady_clock::now();
    modmul_gpu( a.data(), b.data(), a.size(), mod, c.data() );
    auto end = std::chrono::steady_clock::now();
    auto el_time = std::chrono::duration_cast<std::chrono::microseconds>( end - begin ).count();

    std::cerr << "Montgomery multiplication (GPU) on " << size << " elements has finished with elapsed time " << el_time << " ms." << std::endl;

    for( unsigned i = 0; i < naive_c.size(); i++ )
    {
        naive_c[i] = a[i] * b[i] % mod;
    }

    for( unsigned i = 0; i < c.size(); i++ )
    {
        ASSERT_EQ(c[i], naive_c[i]);
    }
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
