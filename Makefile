# Makefile for ScopeCpp

CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall
INCLUDES = -I./include
LDFLAGS = -fopenmp

# 核心源文件（不含 main 入口）
CORE_SRCS = src/Algorithm.cpp \
            src/UniversalData.cpp \
            src/utilities.cpp \
            src/models.cpp \
            src/main.cpp

# 示例源文件
EXAMPLE_SRC = src/example.cpp

# Object files
CORE_OBJS = $(CORE_SRCS:.cpp=.o)
EXAMPLE_OBJ = $(EXAMPLE_SRC:.cpp=.o)

# Target executable
TARGET = scope

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(CORE_OBJS) $(EXAMPLE_OBJ)
	$(CXX) $(CORE_OBJS) $(EXAMPLE_OBJ) -o $(TARGET) $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(CORE_OBJS) $(EXAMPLE_OBJ) $(TARGET)
	rm -f scope.log

# Clean all generated files
distclean: clean
	rm -f *~ src/*~

# Install (optional)
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin"
	@install -m 0755 $(TARGET) /usr/local/bin/

# Run example
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean distclean install run
