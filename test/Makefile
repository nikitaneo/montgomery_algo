CC=/usr/local/cuda-8.0/bin/nvcc
CFLAGS=-std=c++11 -O3 -m64 -ccbin /usr/bin/g++-4.8
LDFLAGS=-lpthread -lgtest -lgmock -lgtest_main -L/usr/lib
INCLUDE=-I /usr/local/include/gtest -I include/ -I ../

all: Test

Test: test.cu
	$(CC) test.cu -o test $(LDFLAGS) $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *.o *~ *.h test