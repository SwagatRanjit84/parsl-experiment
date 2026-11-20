#Compiler gcc 16
CC = gcc-16

#Setting compiler flags for OpenMP
CFLAGS = -Wall -O2 -fopenmp #Enable common compiler warnings and optimization level 2
LDFLAGS = -fopenmp #Setting linker flags

#Setting the name of final executable file
TARGET = prefix_sum_test 

# Setting the object files required to build executable file
OBJS = prefix_sum_test.o prefix_sum_recur.o prefix_sum_tree.o

# Default target: build the executable
all: $(TARGET)

# Linking all object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compiling prefix_sum_test.c into an object file
prefix_sum_test.o: prefix_sum_test.c prefix_sum.h
	$(CC) $(CFLAGS) -c prefix_sum_test.c -o prefix_sum_test.o

# Compiling prefix_sum_recur.c into an object file
prefix_sum_recur.o: prefix_sum_recur.c prefix_sum.h
	$(CC) $(CFLAGS) -c prefix_sum_recur.c -o prefix_sum_recur.o

# Compiling prefix_sum_tree.c into an object file
prefix_sum_tree.o: prefix_sum_tree.c prefix_sum.h
	$(CC) $(CFLAGS) -c prefix_sum_tree.c -o prefix_sum_tree.o

# Removing generated object files, executable, and data files
clean:
	rm -f $(OBJS) $(TARGET) *.dat

.PHONY: all clean
 