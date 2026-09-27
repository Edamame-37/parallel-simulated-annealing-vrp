# Parallel Simulated Annealing CVRP (CUDA)

Implementasi algoritma Simulated Annealing untuk Capacitated Vehicle Routing Problem (CVRP) menggunakan komputasi paralel GPU (CUDA).
Program ini awalnya menggunakan MPI dan sekarang dimigrasi menjadi CUDA.

## Struktur Direktori

- `src/`: Berisi source code CUDA (`main.cu`, `vrp_sa.cu`, `vrp_utils.cu`)
- `include/`: Berisi file header (`vrp.h`)
- `data/`: Dataset VRP
- `mpi-basic/`: Implementasi versi MPI yang asli

## Persyaratan
- NVIDIA GPU dengan driver yang mendukung CUDA
- CUDA Toolkit (termasuk `nvcc`)

## Kompilasi

Kompilasi secara manual menggunakan `nvcc` (jika menggunakan Windows PowerShell):
```powershell
nvcc -O3 -Xcompiler -Wall -I./include src/main.cu src/vrp_sa.cu src/vrp_utils.cu -o vrp_sa_cuda.exe
```
Atau gunakan `make` jika berada di lingkungan Linux/WSL:
```bash
make
```

## Cara Menjalankan Eksekusi

Jalankan program dengan meneruskan argumen berikut:
```bash
./vrp_sa_cuda <dataset> <T0> <Alpha> <Tmin> <Iterations> <Num_Blocks> <Threads_Per_Block>
```

Contoh eksekusi (Windows):
```powershell
.\vrp_sa_cuda.exe data/dataset_medium.txt 1000.0 0.99 0.1 1000 64 256
```
