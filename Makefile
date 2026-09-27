NVCC = nvcc
CFLAGS = -O3 -Xcompiler -Wall
INCLUDES = -I./include

TARGET = vrp_sa_cuda

SRC = src/main.cu src/vrp_sa.cu src/vrp_utils.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
