# ==============================================================================
# Makefile: Massively Parallel Simulated Annealing CVRP (NVIDIA CUDA)
# ==============================================================================

# Compiler CUDA & Flag Kompilasi
NVCC        := nvcc
NVCCFLAGS   := -O3 -std=c++14 -Iinclude -Xcompiler -Wall

# Nama Executable Output
TARGET      := vrp_sa_cuda

# Direktori Proyek
SRCDIR      := src
INCDIR      := include
BUILDDIR    := build

# Otomatisasi Pencarian File Sumber (.cu) dan File Objek (.o)
SOURCES     := $(wildcard $(SRCDIR)/*.cu)
OBJECTS     := $(patsubst $(SRCDIR)/%.cu, $(BUILDDIR)/%.o, $(SOURCES))

# Target Utama
.PHONY: all clean run

all: $(TARGET)

# Linking Executable
$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
	@echo "------------------------------------------------------------------"
	@echo "[SUCCESS] Kompilasi berhasil! Executable: ./$(TARGET)"
	@echo "------------------------------------------------------------------"

# Kompilasi File .cu menjadi File Objek .o
$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Membuat Direktori Build
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# Pembersihan File Hasil Kompilasi
clean:
	rm -rf $(BUILDDIR) $(TARGET)
	@echo "[CLEAN] File biner dan objek berhasil dihapus."

# Menjalankan Program dengan Konfigurasi Default
run-small: $(TARGET)
	./$(TARGET) data/dataset_small.txt 1000.0 0.995 0.001 150 16 128


run-medium: $(TARGET)
	./$(TARGET) data/dataset_medium.txt 1000.0 0.995 0.001 150 16 128

