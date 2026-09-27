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

Gunakan `make` untuk melakukan kompilasi proyek:
```bash
make
```

Ini akan menghasilkan executable bernama `vrp_sa_cuda`.

## Cara Menjalankan Eksekusi

Jalankan program dengan meneruskan argumen berikut:
```bash
./vrp_sa_cuda <dataset> <T0> <Alpha> <Tmin> <Iterations> <Num_Blocks> <Threads_Per_Block>
```

Contoh eksekusi:
```bash
./vrp_sa_cuda data/dataset_medium.txt 1000.0 0.99 0.1 1000 64 256
```
