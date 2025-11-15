# Makefile for CUDA C project

# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_50 -std=c++11
TARGET = vector_add
SOURCE = vector_add.cu

# Default target
all: $(TARGET)

# Build the CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Compile the CUDA program"
	@echo "  make run      - Compile and run the program"
	@echo "  make clean    - Remove compiled files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - NVIDIA GPU with CUDA support"
	@echo "  - CUDA Toolkit installed (nvcc compiler)"
	@echo "  - Compatible GPU driver"

.PHONY: all run clean help
