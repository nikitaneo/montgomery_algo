CC=g++
CFLAGS=-std=c++11 -g
INCLUDE=include/uint128_t
OBJS=objs
SOURCES=src
BUILD=build

all: $(OBJS)/main.o $(OBJS)/uint128_t.o
	$(CC) $(OBJS)/main.o $(OBJS)/uint128_t.o -o $(BUILD)/mult.out $(LIBS) $(CFLAGS)

$(OBJS)/main.o: $(SOURCES)/main.cpp
	$(CC) -c $(SOURCES)/main.cpp -o $(OBJS)/main.o -I $(INCLUDE) $(CFLAGS)

$(OBJS)/uint128_t.o: $(SOURCES)/uint128_t.cpp
	$(CC) -c $(SOURCES)/uint128_t.cpp -o $(OBJS)/uint128_t.o -I $(INCLUDE) $(CFLAGS)

clean:
	rm -rf $(BUILD)/*.out
	rm -rf $(OBJS)/*.o