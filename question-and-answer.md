# Pertanyaan dan Jawaban Teknis CUDA (CPL06 / CPL07)

## 1. Mengidentifikasi pekerjaan yang dapat dilakukan secara bersamaan
**Bagian 1, Pertanyaan ke-1:** *"Bagian mana dari algoritma Anda yang dapat dijalankan secara bersamaan, dan bagian mana yang harus tetap berurutan? Gambarkan alur dari pembacaan input sampai keluaran akhir, lalu tunjukkan dependensi yang menjadi alasan setiap keputusan."*

**Jawaban:**
Pada program optimasi CVRP menggunakan Simulated Annealing (SA) versi CUDA, pekerjaan yang dapat dijalankan secara **bersamaan (paralel)** adalah **proses pencarian rute terbaik (eksplorasi SA) itu sendiri**. Setiap *thread* pada GPU menjalankan instans SA yang terpisah secara penuh, mulai dari inisialisasi rute acak, penelusuran *neighborhood*, hingga evaluasi solusi iteratif. Karena sifat heuristik pencarian acak, masing-masing proses ini saling independen dan tidak memiliki dependensi data antar-thread; setiap thread mencari di ruang solusi yang berbeda karena status pengacakannya (seed *cuRAND*) diatur unik.

Namun, bagian yang harus tetap **berurutan (sekuensial)** meliputi:
1. **Pembacaan input dari disk:** Operasi I/O (membaca file dataset) dilakukan secara sekuensial oleh *Host* (CPU) sebelum fase paralelisme dimulai.
2. **Transfer data Host-to-Device:** Data koordinat/nodes harus disalin penuh ke memori GPU terlebih dahulu sebelum *kernel* diluncurkan karena GPU membutuhkan titik referensi global sebagai prasyarat.
3. **Looping Penurunan Suhu dan Iterasi SA (di dalam tiap thread):** Proses SA pada satu rentang suhu mutlak bergantung pada hasil dari iterasi suhu sebelumnya (*Markov Chain*). Tidak mungkin menjalankan iterasi suhu $T_2$ secara bersamaan dengan $T_1$ karena $T_2$ memerlukan rute *current_solution* yang ditinggalkan oleh tahapan akhir iterasi $T_1$.
4. **Reduksi Global Pencarian Pemenang:** Setelah seluruh *thread* GPU selesai, komputasi mencari siapa di antara ribuan *thread* tersebut yang memiliki rekor jarak absolut terpendek (reduksi minimum) dilakukan secara linear berurutan di CPU. Dependensinya jelas: reduksi final tidak bisa berjalan jika thread GPU belum semuanya selesai mengirim balik solusi.

---

## 2. Mempartisi dan memetakan pekerjaan ke pemroses
**Bagian 2, Pertanyaan ke-1:** *"Bagaimana pekerjaan dibagi dan dipetakan pada versi MPI serta versi CUDA Anda? Tunjukkan hubungan antara indeks data global, rank MPI, indeks lokal, serta blockIdx dan threadIdx pada implementasi masing-masing."*

**Jawaban (Fokus CUDA):**
Pekerjaan dipartisi menggunakan pola eksekusi *Task Parallelism* melalui pemanfaatan struktur *Grid* satu dimensi (1D) yang menaungi *Blocks* dan *Threads*. Karena ini adalah pencarian ruang heuristik yang dikerjakan masif-serentak (bukan pembagian array data secara tradisional), pemetaan didasarkan sepenuhnya dari *total thread* alokasi pencarian pengguna.

Hubungan penentuan **ID/indeks global unik** dari setiap thread pekerja (penjelajah SA) diturunkan dari struktur hierarkis SM (Streaming Multiprocessor) CUDA dengan rumusan linier:
```c
int id = threadIdx.x + blockIdx.x * blockDim.x;
```
- `threadIdx.x`: Merupakan indeks lokal thread di dalam sebuah Block. Rentangnya dari $0$ hingga $(blockDim.x - 1)$.
- `blockIdx.x`: Merupakan indeks/koordinat dari Block tersebut di dalam cakupan Grid.
- `blockDim.x`: Merupakan ukuran/jumlah total *threads* yang didefinisikan per *block* (misalnya 256).

Dengan `id` global linier tersebut, thread mengetahui:
1. Titik akses ke status *seed* generator acaknya sendiri (*cuRAND state*) dari `state[id]`.
2. Indeks penulisan memori persis (bebas interupsi/tabrakan) dari memori *array output* solusi terbaiknya ke `d_best_solutions[id]`.

---

## 3. Mendistribusikan input, output, dan data antara
**Bagian 3, Pertanyaan ke-4:** *"Pada implementasi CUDA, data mana yang perlu ditransfer dari host ke device, data mana yang dapat tetap berada di device, dan data mana yang perlu dikembalikan ke host? Jelaskan juga alasan penggunaan global memory, shared memory, atau constant memory apabila digunakan."*

**Jawaban:**
1. **Ditransfer dari Host ke Device (H2D):**
   - Array `Node` yang berisi letak koordinat pelanggan, titik berat *demand* muatan, dan depot. Ini karena pembacaan data awal (file .txt) hanya bisa dilakukan oleh modul CPU, sehingga disalin menggunakan `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
2. **Tetap berada di Device (Diciptakan dan musnah di Device):**
   - Array status (*state*) dari *random number generator* (`cuRAND state`). Array ini dialokasikan di VRAM (`cudaMalloc`) dan dimodifikasi langsung oleh *kernel setup*. Tidak ada nilai intrinsik darinya yang perlu dibaca CPU.
   - Variabel dan *struct* rute sementara (seperti `current_sol`, `next_sol`) di dalam *kernel loop*. Ini sepenuhnya dialokasikan sebagai data lokal yang umumnya dikompilasi masuk ke dalam alokasi **Register** atau **Local Memory** setiap *thread* karena sifat pakainya yang sementara (*scratchpad*) dengan performa operasi R/W tertinggi.
3. **Dikembalikan ke Host (D2H):**
   - Array raksasa (tergantung *total threads*) `d_best_solutions` yang merupakan kumpulan rute hasil pencarian terbaik masing-masing pekerja GPU. Data ini diambil melalui `cudaMemcpyDeviceToHost` agar CPU dapat melakukan komputasi penyeleksian (reduksi) terakhir terhadap siapa yang menang.

*Pemilihan Global Memory:* Array `nodes` ditaruh di dalam **Global Memory** karena kapasitas datanya besar dan dapat menampung ribuan node dataset VRP. Karena elemen ini sifatnya *read-only* saat komputasi berlangsung (dataset peta logistik tidak berubah) pola *cache* L1/L2 dari GPU otomatis akan menangani beban permintaannya *(broadcasting cache)* tanpa masalah besar.

---

## 4. Mengoordinasikan akses data untuk menghindari konflik
**Bagian 4, Pertanyaan ke-4:** *"Apabila banyak pekerja berkontribusi terhadap satu hasil, seperti jumlah total, histogram, atau nilai maksimum, bagaimana konflik pembaruan dicegah? Bandingkan pilihan akumulasi privat, reduksi bertahap, atau operasi atomik berdasarkan kebenaran, ketelitian, dan overhead."*

**Jawaban:**
Pada solusi VRP ini, banyak pekerja menemukan solusinya masing-masing untuk dikontribusikan menuju satu temuan akhir: **Rute Minimum Global**. Jika seluruh thread berebut menimpa satu variabel tunggal *GlobalBest* setiap kali menemukan rute baru, akan timbul tabrakan baca/tulis *(race condition)* parah.

Program kita menghindari konflik melalui teknik pendekatan **Akumulasi Privat (Private Output Array)** tanpa persilangan *write access*.
Setiap thread dibiarkan menulis murni pada "kotak penyimpanan" atau offset elemen miliknya sendiri melalui referensi `d_best_solutions[id]`. Setelah seluruh array global ini tuntas, perangkuman (*Reduction* untuk mendapatkan nilai terendah) sepenuhnya dikerjakan satu pihak saja yaitu Host (CPU) pada tahap akhir menggunakan fungsi loop linear berurutan.

Perbandingan Teknik:
- **Operasi Atomik (Atomic Min):** Bawaan GPU menjamin kebenaran 100% dan bebas konflik. Masalahnya, operasi ini *sangat mematikan overhead performa* jika diadu oleh puluhan ribu thread pada satu alamat memori persis, dan lebih menyulitkan karena *struct Solution VRP* berisikan array rute pelanggan dan bertipe floating-point jarak (*double*).
- **Reduksi Bertahap (*Tree-Based Reduction* pada Shared Memory GPU):** Sangat disarankan untuk kinerja puncak. Overhead DRAM sangat rendah karena perhitungan diciutkan dulu per *block* oleh *shared memory* lokal, lalu sisa elemen kecil dilimpahkan lagi untuk reduksi. Namun kompleksitas kode menyalin struktur data besar (array tour `Solution`) melintasi *shared memory* membutuhkan kalkulasi *stride/pitch* yang tinggi tingkat kesulitannya.
- **Akumulasi Privat lalu Reduksi di Host (Sesuai Program):** Kebenaran tinggi (100% tanpa risiko tertimpa), namun *overhead memory array* linear $(O(N))$ cukup gemuk (GPU butuh VRAM sejumlah ukuran Struct $\times$ total thread). Pendekatan ini merupakan *trade-off* terbaik untuk kemudahan (*safest implementation*) agar tahap komputasi asinkron GPU tetap melaju maksimal tanpa titik tunda sinkronisasi.

---

## 5. Menjamin urutan pekerjaan melalui sinkronisasi
**Bagian 5, Pertanyaan ke-5:** *"Bagaimana Anda menjamin urutan transfer host-to-device, eksekusi kernel, transfer device-to-host, dan penggunaan hasil oleh CPU? Tunjukkan dependensi yang digunakan pada satu stream maupun beberapa stream, termasuk peran event atau penantian host apabila diperlukan."*

**Jawaban:**
Aliran urutan eksekusi (*pipeline dependency*) sistem berbasis CUDA dijamin kebenarannya secara sekuensial dengan perpaduan asinkronisasi kernel, barikade sinkronisasi eksplisit GPU, serta blokade sinkron memori melalui satu aliran perintah utama (*Stream 0 / Default*).

1. **Host-to-Device (H2D) $\rightarrow$ Kernel Execution:** 
   Proses penyalinan awal memanggil `cudaMemcpy(..., cudaMemcpyHostToDevice)`. Di aliran standar, fungsi API ini bersifat **Synchronous** terhadap pemanggil (CPU). Artinya, CPU tidak akan mengalir dan memanggil peluncuran rutin *kernel* (operator pembuka kerja GPU) sebelum proses transmisi data dataset 100% dijamin rampung tertanam ke VRAM.
2. **Kernel Execution $\rightarrow$ Device-to-Host (D2H):**
   Peluncuran kernel `run_simulated_annealing_kernel<<<...>>>` dikerjakan GPU secara bebas dan *Asynchronous* (CPU langsung lolos seketika mengeksekusi sintaks di bawahnya tanpa menunggu hasil). Untuk mencegah CPU nekat mencoba me-return memori (`cudaMemcpy` D2H) sedangkan pekerja GPU masih sibuk mutasi genetik rute SA, kita wajib meletakkan palang peringatan eksplisit `cudaDeviceSynchronize()`. Ini bertindak sebagai tameng blocking yang membekukan aliran Host sampai sinyal *all-kernel-done* dari antrean eksekusi Device keluar. Barulah setelah itu operasi *Memcpy D2H* aman menarik *array* berisi jawaban ke CPU.
3. **Penyelarasan Presisi Waktu via Event Marker:**
   Kita menyelipkan dependensi pengukuran waktu murni GPU dengan fungsi `cudaEventRecord`. Tanda "Start" diletakkan sebelum kernel *wrapper* dipanggil dan "Stop" setelah itu. CPU wajib dicegat oleh `cudaEventSynchronize(stop)` sebelum mengekstrak milidetik, mencegah pencetakan estimasi dini yang menyesatkan sementara sinyal GPU di *stream* antrean belum genap merambah pita rekam *stop* di *hardware* GPU-nya.
