CC=g++
CFLAGS=-std=c++11 -g -G
INCLUDE=include
OBJS=objs
SOURCES=src
BUILD=build

all: $(OBJS)/main.o
	$(CC) $(OBJS)/main.o -o $(BUILD)/mult.out $(LIBS) $(CFLAGS)

$(OBJS)/main.o: $(SOURCES)/main.cu
	$(CC) -c $(SOURCES)/main.cu -o $(OBJS)/main.o -I $(INCLUDE) $(CFLAGS)

clean:
	rm -rf $(BUILD)/*.out
	rm -rf $(OBJS)/*.o