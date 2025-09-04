# Compiler: Use hipcc for DCU code
CXX = hipcc

# Compiler flags
CXXFLAGS = -I./inc -O3

# Linker flags for DCU: Link rocblas and rocsparse
LDFLAGS = -lrocblas -lrocsparse

# Source files for DCU version
SRCS = src/main.cpp src/gmres_dcu.cpp # 注意这里文件名变了

# Object files
OBJS = $(patsubst src/%.cpp, obj/%.o, $(SRCS))

# Executable name
TARGET = gmres

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile source files to object files
obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f obj/*.o $(TARGET) gmres_time.txt *.out *.err

.PHONY: all clean