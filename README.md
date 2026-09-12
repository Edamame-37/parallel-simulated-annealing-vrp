# Parallel Simulated Annealing untuk Optimasi Vehicle Routing Problem (VRP) Menggunakan MPI

Proyek ini mengimplementasikan algoritma metaheuristik *Simulated Annealing* (SA) secara paralel menggunakan *Message Passing Interface* (MPI) untuk memecahkan *Capacitated Vehicle Routing Problem* (CVRP).

Pendekatan paralel yang digunakan adalah **Multiple Independent Markov Chains**, di mana setiap node/prosesor (Rank) mengeksplorasi ruang solusi yang benar-benar berbeda secara mandiri, lalu hasil terbaik dari seluruh node dikumpulkan untuk mendapatkan rute global paling optimal.

---

## Konsep Algoritma

### 1. Vehicle Routing Problem (VRP)

VRP adalah masalah optimasi kombinatorial untuk menemukan rute pengiriman barang paling efisien dari sebuah depot ke sekumpulan pelanggan. Setiap kendaraan memiliki kapasitas maksimum. Tujuan utamanya adalah **meminimalkan total jarak tempuh** tanpa melanggar batas kapasitas kendaraan.

### 2. Simulated Annealing (SA)

SA adalah algoritma heuristik yang terinspirasi dari proses pendinginan logam. Algoritma ini mencari rute terbaik dengan cara:

* Memulai dengan rute acak pada suhu tinggi (T).
* Melakukan modifikasi rute (menukar posisi pelanggan secara acak).
* Jika rute baru lebih pendek, rute tersebut **diterima**.
* Jika rute baru lebih panjang, rute tersebut **masih bisa diterima** dengan probabilitas $e^{-\Delta/T}$ (Probabilitas Boltzmann). Ini mencegah algoritma terjebak di *Local Optima*.
* Suhu (T) diturunkan secara perlahan. Semakin dingin, sistem semakin sulit menerima rute yang lebih buruk, hingga akhirnya membeku di rute terbaik (*Global/Local Optima*).

---

## Cara Kerja Program (Pipeline MPI)

Program berjalan dalam 4 fase utama:

1. **Fase Inisialisasi (Master Node - Rank 0)**
* Master membaca dataset pelanggan (koordinat X, Y, dan permintaan barang) dari file eksternal (misal: format TXT atau TSPLIB).
* Master mendistribusikan konfigurasi awal (jumlah kota, kapasitas kendaraan, parameter SA) ke semua *Worker* menggunakan `MPI_Bcast`.


2. **Fase Eksekusi Paralel (Semua Rank)**
* Setiap Rank menginisialisasi *Random Seed* yang berbeda berdasarkan ID Rank-nya (`srand(time(NULL) + rank)`). Hal ini memastikan setiap prosesor memulai pencarian dari rute awal yang berbeda.
* Setiap Rank menjalankan proses *Simulated Annealing* secara penuh secara lokal (pendinginan dari suhu awal hingga suhu akhir).
* Selama proses ini, **tidak ada komunikasi antar node** (*Embarrassingly Parallel*), sehingga meniadakan *overhead* jaringan dan memaksimalkan kinerja komputasi CPU.


3. **Fase Reduksi (Pengumpulan Hasil)**
* Setelah pendinginan selesai, setiap Rank memiliki jarak terpendek (Best Local Distance) masing-masing.
* Menggunakan `MPI_Reduce` dengan operasi `MPI_MINLOC`, Master mencari **Jarak Terpendek Global** sekaligus mengetahui **Rank mana** yang memilikinya.


4. **Fase Rekonstruksi & Output (Master Node)**
* Jika Master bukan Rank pemenang, Master akan meminta array urutan rute lengkap dari Rank pemenang menggunakan `MPI_Recv`.
* Rank pemenang mengirim array rutenya menggunakan `MPI_Send`.
* Master mencetak urutan rute terbaik, total jarak, dan waktu eksekusi ke layar atau file *log*.



---

## Input dan Output

* **Input:**
* File dataset berisi informasi node (Depot dan Pelanggan), mencakup: ID, Koordinat X, Koordinat Y, dan *Demand* (permintaan).
* Parameter algoritma (dapat di-set di argumen terminal): Suhu Awal ($T_0$), Laju Pendinginan ($\alpha$), Batas Suhu ($T_{min}$), dan Kapasitas Kendaraan.


* **Output:**
* Total jarak tempuh paling minimal.
* Array/urutan kunjungan kendaraan (misal: `Depot -> P1 -> P4 -> Depot -> P2 -> P3 -> Depot`).
* Waktu komputasi yang dihabiskan.



---

## Perencanaan Pengembangan (Roadmap)

* [ ] **Tahap 1: Persiapan Serial (Tanpa MPI)**
* Membuat *struct* data untuk merepresentasikan Kota dan Kendaraan.
* Membuat fungsi penghitung jarak Euclidean antar dua titik.
* Menulis fungsi evaluasi rute (termasuk pengecekan kapasitas kendaraan).
* Implementasi *loop* algoritma *Simulated Annealing* dasar.


* [ ] **Tahap 2: Integrasi MPI (Distribusi Data)**
* Inisialisasi `MPI_Init`.
* Master membaca file dataset dan mem-parsing datanya ke dalam array.
* Implementasi `MPI_Bcast` untuk menyalin array dataset dari Master ke semua memori *Worker*.


* [ ] **Tahap 3: Paralelisasi SA & Reduksi**
* Pemisahan *Random Seed* berdasarkan ID Rank.
* Eksekusi SA pada masing-masing Rank.
* Pengumpulan nilai jarak terpendek menggunakan `MPI_Reduce` (`MPI_MINLOC`).
* Transfer array rute dari pemenang ke Master menggunakan `MPI_Send` / `MPI_Recv`.


* [ ] **Tahap 4: Pengujian & Pembuatan Laporan**
* Uji coba program dengan dataset kecil (10-20 kota) dan dataset besar (>100 kota).
* Mengukur waktu eksekusi menggunakan `MPI_Wtime()`.
* Membandingkan metrik waktu dan hasil optimasi antara 1 Core, 2 Core, dan 4 Core (Menghitung *Speedup*).



---

## Panduan Kompilasi dan Eksekusi

Pastikan Anda memiliki *compiler* C/C++ dan pustaka MPI (seperti OpenMPI atau MPICH) terinstal di sistem Anda (misal: WSL/Linux).

**1. Kompilasi Program**

```bash
make

```

*(Catatan: flag `-lm` digunakan jika Anda menggunakan library `<math.h>` untuk perhitungan eksponensial Boltzmann dan jarak Euclidean).*

**2. Menjalankan Program**

```bash
# Menjalankan dengan 4 prosesor (dataset kecil)
make run-small

```

```bash
# Menjalankan dengan 4 prosesor (dataset besar)
make run-medium

```


## 📌 Gambaran Umum & Konsep

**Vehicle Routing Problem (VRP)** bertujuan untuk menemukan rute paling optimal bagi sejumlah armada kendaraan berkapasitas terbatas dalam melayani sekumpulan pelanggan dari satu titik pusat (depot). 

Pendekatan paralel yang digunakan adalah **Island Model / Independent Search**, di mana setiap prosesor (MPI Rank) menjalankan simulasi pencarian rute secara mandiri dengan *random seed* dan skema pendinginan yang bervariasi. Pendekatan ini secara drastis mengurangi risiko terjebak dalam solusi *local optima*.

```text=
SEBELUM OPTIMASI (Acak):
    (C3)      (C4)
      \       /
  (C2)--[DEPOT]--(C5)   -> Jarak: 245 km (Tumpang Tindih)
      /       \
    (C1)      (C6)

SESUDAH OPTIMASI (Efisien):
    (C3)------(C4)
     /          \
  (C2)  [DEPOT]  (C5)   -> Jarak: 112 km (Rute Optimal)
     \    /   \    /
     (C1)       (C6)

```
## 🎨 Diagram Visualisasi Cara Kerja MPI

### 1. Alur Arsitektur Paralel Master-Worker

+-------------------------------------------------------+
|                    MASTER (Rank 0)                    |
| 1. Membaca dataset lokasi & kapasitas armada.         |
| 2. Distribusi data ke seluruh Worker via MPI_Bcast(). |
+-------------------------------------------------------+
                           |
                           v (MPI_Bcast)
     +---------------------+---------------------+
     |                     |                     |
     v                     v                     v
+------------+        +------------+        +------------+
|  WORKER 1  |        |  WORKER 2  |        |  WORKER N  |
|  (Rank 1)  |        |  (Rank 2)  |        |  (Rank N)  |
+------------+        +------------+        +------------+
| Seed: 101  |        | Seed: 202  |        | Seed: N0N  |
| T: 1000°C  |        | T: 1200°C  |        | T: 900°C   |
| Alpha: 0.95|        | Alpha: 0.90|        | Alpha: 0.98|
+------------+        +------------+        +------------+
     |                     |                     |
     v                     v                     v
[  SA Loop  ]         [  SA Loop  ]         [  SA Loop  ]
     |                     |                     |
     v                     v                     v
(Best Sol 1)          (Best Sol 2)          (Best Sol N)
     |                     |                     |
     +---------------------+---------------------+
                           |
                           v (MPI_Reduce: MPI_MINLOC)
+-------------------------------------------------------+
|                    MASTER (Rank 0)                    |
| 3. Ambil solusi terbaik (Global Minimum Distance).    |
| 4. Cetak Rute Optimal & Statistik Performa.           |
+-------------------------------------------------------+


### 2. Alur Pencarian Solusi di Setiap Node (Simulated Annealing Loop)

Setiap Worker menjalankan iterasi berikut tanpa saling mengganggu (*zero communication overhead* selama fase komputasi):

                   +------------------------+
                   | Formulasi Solusi Awal  |
                   |  (Generasi Rute Acak)  |
                   +------------------------+
                               |
                               v
         +--------------------------------------------+
         | Mutasi Rute (Swap / 2-Opt / Reinsert Node) | <---------+
         +--------------------------------------------+           |
                               |                                  |
                               v                                  |
                  +--------------------------+                    |
                  |   Hitung ΔCost (Jarak)   |                    |
                  +--------------------------+                    |
                               |                                  |
               +---------------+---------------+                  |
               |                               |                  |
       [ΔCost < 0]                     [ΔCost >= 0]               |
      (Lebih Bagus)                  (Lebih Buruk)                |
               |                               |                  |
               v                               v                  |
       +---------------+               +---------------+          |
       | Terima Solusi |               |  Hitung P =   |          |
       +---------------+               | exp(-ΔCost/T) |          |
               |                       +---------------+          |
               |                               |                  |
               |                     +---------+---------+        |
               |                     |                   |        |
               |              [random() < P]    [random() >= P]   |
               |               (Terima Acak)       (Tolak)        |
               |                     |                   |        |
               |                     v                   v        |
               +---------------------+-------------------+        |
                                     |                            |
                                     v                            |
                         +-----------------------+                |
                         | Diturunkan Suhu (T)   |                |
                         |  T = T * Alpha        |                |
                         +-----------------------+                |
                                     |                            |
                             [Belum Selesai]                      |
                                     |                            |
                                     +----------------------------+
                                     |
                                [Suhu T Min]
                                     |
                                     v
                         +-----------------------+
                         | Kirim Solusi Terbaik  |
                         |   Lokal ke Master     |
                         +-----------------------+

## 🛠️ Prasyarat System

* **Compiler C/C++:** `gcc` / `g++` (versi 9.0 atau yang lebih baru)
* **MPI Library:** `OpenMPI` (v4.0+) atau `MPICH`
* **Build System:** `Make`

---

## 📁 Struktur Direktori

```text
.
├── data/
│   └── dataset_medium.txt       # Dataset koordinat pelanggan & kapasitas
|   └── dataset_small.txt       # Dataset koordinat pelanggan & kapasitas
├── src/
│   ├── main.c                 # Logika utama MPI & Komunikasi Node
│   ├── vrp_sa.c               # Algoritma Simulated Annealing & Mutasi Rute
│   └── vrp_utils.c               # Definisi Struktur Data & Header Function
├── Makefile                   # Skrip otomatisasi kompilasi
└── README.md                  # Dokumentasi proyek