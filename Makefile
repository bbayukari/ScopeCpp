# Makefile for ScopeCpp

CXX = g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall
INCLUDES = -I./include
LDFLAGS = -fopenmp

# Source files
SRCS = src/Algorithm.cpp \
       src/UniversalData.cpp \
       src/utilities.cpp \
       src/models.cpp \
       src/main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = scope

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)
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
