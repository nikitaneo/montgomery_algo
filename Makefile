CC=/usr/local/cuda-8.0/bin/nvcc
CFLAGS=-std=c++11 -O3 -m64 -ccbin /usr/bin/g++-4.8
INCLUDE=include
OBJS=objs
SOURCES=src
BUILD=build

all: $(OBJS)/main.o
	$(CC) $(OBJS)/main.o -o $(BUILD)/mult.out $(LIBS) $(CFLAGS)

$(OBJS)/main.o: $(SOURCES)/main.cu include/uint128_t.h
	$(CC) -c $(SOURCES)/main.cu -o $(OBJS)/main.o -I $(INCLUDE) $(CFLAGS)

clean:
	rm -rf $(BUILD)/*.out
	rm -rf $(OBJS)/*.o
