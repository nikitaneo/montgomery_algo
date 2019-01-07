#include "fastmul.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <uint128_t.h>
#include <vector>
#include <iostream>
#include <chrono>

class FastMultTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        a = std::vector<uint128_t>( size );
        b = std::vector<uint128_t>( size );
        c = std::vector<uint128_t>( size );

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937_64 gen( seed );
        std::uniform_int_distribution<uint64_t> dist( 1, uint64_t(-1) );
        for( unsigned i = 0; i < size; i++ )
        {
            a[i] = dist( gen );
            b[i] = dist( gen );
        }
        
        for( unsigned i = 0; i < c.size(); i++ )
        {
            c[i] = a[i] * b[i] % mod;
        }
    }

    unsigned size{ 1024 * 1024 };
    uint128_t mod{ 14797879391564958223ULL };
    std::vector<uint128_t> a;
    std::vector<uint128_t> b;
    std::vector<uint128_t> c;
};

TEST_F(FastMultTest, CPU)
{
    std::vector<uint128_t> fast_c( size );

    auto begin = std::chrono::steady_clock::now();
    modmul_cpu( a.data(), b.data(), a.size(), mod, fast_c.data() );
    auto end = std::chrono::steady_clock::now();
    auto el_time = std::chrono::duration_cast<std::chrono::microseconds>( end - begin ).count();

    std::cerr << "Montgomery multiplication (CPU) on " << size << " elements has finished with elapsed time " << el_time << " ms." << std::endl;

    for( unsigned i = 0; i < c.size(); i++ )
    {
        ASSERT_EQ(c[i], fast_c[i]);
    }
}

TEST_F(FastMultTest, GPU)
{
    std::vector<uint128_t> fast_c( size );

    auto begin = std::chrono::steady_clock::now();
    modmul_gpu( a.data(), b.data(), a.size(), mod, fast_c.data() );
    auto end = std::chrono::steady_clock::now();
    auto el_time = std::chrono::duration_cast<std::chrono::microseconds>( end - begin ).count();

    std::cerr << "Montgomery multiplication (GPU) on " << size << " elements has finished with elapsed time " << el_time << " ms." << std::endl;

    for( unsigned i = 0; i < c.size(); i++ )
    {
        ASSERT_EQ(c[i], fast_c[i]);
    }
}

int main(int argc, char ** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
