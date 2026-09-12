# Makefile untuk Parallel Simulated Annealing CVRP Menggunakan MPI (Modular)

# Compiler dan flags
CC = mpicc
CFLAGS = -Wall -O2 -Iinclude
LDFLAGS = -lm

# Direktori dan berkas sumber
SRC_DIR = src
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/vrp_utils.c $(SRC_DIR)/vrp_sa.c
HEADERS = include/vrp.h

# Nama target binary
TARGET = vrp_sa

# Dataset default
DATA_SMALL = data/dataset_small.txt
DATA_MEDIUM = data/dataset_medium.txt

.PHONY: all clean run-small run-medium help

# Target default: kompilasi seluruh modul
all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Menjalankan dengan dataset kecil menggunakan 4 prosesor
run-small: $(TARGET)
	mpirun -np 4 ./$(TARGET) $(DATA_SMALL)

# Menjalankan dengan dataset medium menggunakan 4 prosesor
run-medium: $(TARGET)
	mpirun -np 4 ./$(TARGET) $(DATA_MEDIUM)

# Membersihkan file binary hasil kompilasi
clean:
	rm -f $(TARGET) $(TARGET).exe

# Menampilkan bantuan penggunaan Makefile
help:
	@echo "Perintah yang tersedia:"
	@echo "  make            - Mengompilasi seluruh modul (mpicc -Iinclude src/*.c)"
	@echo "  make run-small  - Menjalankan 4 prosesor pada dataset kecil (15 pelanggan)"
	@echo "  make run-medium - Menjalankan 4 prosesor pada dataset menengah (50 pelanggan)"
	@echo "  make clean      - Menghapus binary keluaran"
