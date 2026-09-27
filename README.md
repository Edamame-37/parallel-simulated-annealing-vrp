# Massively Parallel Simulated Annealing untuk CVRP menggunakan NVIDIA CUDA

Proyek ini mengimplementasikan algoritma **Simulated Annealing (SA) Massively Parallel** untuk menyelesaikan masalah **Capacitated Vehicle Routing Problem (CVRP)** memanfaatkan akselerasi GPU NVIDIA melalui arsitektur CUDA.

## 🚀 Fitur Utama & Arsitektur GPU
- **Massively Parallel Execution**: Masing-masing thread CUDA mengeksekusi 1 rantai pencarian (*search chain*) Simulated Annealing secara independen dengan *random seed* unik.
- **cuRAND Integration**: Menggunakan pustaka `cuRAND` untuk pembangkitan angka acak paralel berkecepatan tinggi per-thread pada VRAM.
- **High-Performance Memory Layout**: Menggunakan struktur data berukuran tetap (`tour[MAX_CUSTOMERS]`) untuk memanfaatkan *Register* & *Thread-Local Memory* GPU secara maksimal tanpa alokasi memori dinamis di dalam kernel.
- **Global Best Reduction**: Menggabungkan hasil solusi lokal dari ribuan thread GPU untuk menemukan solusi optimal global dengan konsumsi waktu minimal.
- **High-Precision Timing**: Pengukuran waktu komputasi GPU presisi tinggi menggunakan `cudaEvent_t`.

---

## 🛠️ Prasyarat Sistem
Pastikan perangkat keras dan perangkat lunak berikut telah terinstal pada sistem Anda:
1. **GPU NVIDIA** dengan *Compute Capability* 3.0 atau lebih baru.
2. **NVIDIA CUDA Toolkit** (versi 10.0 atau yang lebih baru, direkomendasikan CUDA 11.x / 12.x).
3. **Compiler `nvcc`** yang sudah terdaftar dalam `PATH` sistem.
4. OS Linux (Ubuntu/Debian) atau Windows (WSL2 / Visual Studio Command Prompt).

---

## 📁 Struktur Direktori
```text
parallel-simulated-annealing-cvrp/
├── data/
│   ├── dataset_medium.txt      # Dataset uji skala menengah
│   └── dataset_small.txt       # Dataset uji skala kecil
├── include/
│   └── vrp.h                   # Header utama, definisi struct, dan prototipe fungsi
├── src/
│   ├── main.cu                 # Entry point, CLI parsing, dan pencetakan hasil
│   ├── vrp_sa.cu               # Kernel CUDA, device functions, dan host wrapper
│   └── vrp_utils.cu            # Reader dataset dan kalkulasi jarak geometri
├── Makefile                    # Skrip kompilasi NVCC
└── README.md                   # Dokumentasi proyek