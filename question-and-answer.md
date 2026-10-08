# Panduan Presentasi Teknis & Demo Eksekusi CUDA (Total Durasi: 10 Menit)
*(Terintegrasi dengan Jawaban Bank Pertanyaan CPL06 / CPL07)*

---

## 1. Pendahuluan & Demo Eksekusi Program [Menit 0:00 - 2:00]

**(Apa yang perlu dijelaskan kepada audiens/penguji saat program dijalankan secara langsung):**
1. **Cara Kompilasi & Parameter Eksekusi:**
   Tunjukkan perintah untuk menjalankan program, misalnya `./vrp_sa data/dataset.txt 1000 0.99 0.001 100 64 256`. Jelaskan secara singkat bahwa kita melempar argumen *hyperparameter* SA (Suhu Awal, Laju Pendinginan, Suhu Minimum, Iterasi) serta *konfigurasi Grid GPU* (Jumlah Block dan Thread per Block).
2. **Proses I/O & Transfer H2D (Host-to-Device):**
   Saat program mulai menampilkan informasi *Dataset* dan *Capacity*, sampaikan bahwa modul CPU (Host) sedang membaca file teks mentah secara sekuensial, dan kemudian langsung memompakan data peta rute tersebut ke memori VRAM GPU.
3. **Eksekusi Paralel (Jantung Komputasi):**
   Saat layar tampak berhenti sejenak (proses menghitung), sampaikan bahwa di dalam cip GPU sedang terjadi ribuan proses penelusuran (Simulated Annealing) yang independen. Apabila kita menset parameter `64 blocks` $\times$ `256 threads`, artinya ada $16.384$ agen heuristik yang mengacak, menghitung, dan mencari rute di ruang solusi secara serentak tanpa saling menunggu.
4. **Hasil Akhir & Waktu Eksekusi (Execution Time):**
   Ketika hasil *Total Distance* dan metrik milidetik (*Execution Time*) dicetak, jelaskan bahwa angka waktu tersebut dicatat langsung oleh lapisan *hardware* GPU menggunakan fitur *CUDA Event Marker*, bukan *timer* CPU biasa yang rentan latensi *system call*. CPU baru mengambil data pemenangnya saja.

---

## 2. Mengidentifikasi Pekerjaan Bersamaan vs Sekuensial [Menit 2:00 - 3:30]
**Bagian 1, Pertanyaan ke-1:** *"Bagian mana dari algoritma Anda yang dapat dijalankan secara bersamaan, dan bagian mana yang harus tetap berurutan? Gambarkan alur dari pembacaan input sampai keluaran akhir, lalu tunjukkan dependensi yang menjadi alasan setiap keputusan."*

**Jawaban:**
*(Diimplementasikan pada file: `src/main.cu` baris ke-22 dan `src/vrp_sa.cu` baris ke-112)*

Pada optimasi CVRP berbasis Simulated Annealing (SA) CUDA, pekerjaan yang **bersamaan (paralel)** adalah **proses pencarian rute terbaik (eksplorasi SA) itu sendiri**. Setiap *thread* pada GPU menjalankan instans SA yang terpisah secara penuh (inisialisasi rute, penelusuran *neighborhood*, hingga evaluasi). Karena sifat heuristik acak, proses ini saling independen; setiap thread mencari di ruang solusi berbeda menggunakan *seed cuRAND* unik.

Namun, bagian yang harus tetap **berurutan (sekuensial)** meliputi:
1. **Pembacaan input dari disk:** Operasi I/O CPU sebelum fase paralelisme dimulai.
2. **Transfer data Host-to-Device:** GPU membutuhkan referensi data titik secara global sebelum *kernel* diluncurkan.
3. **Looping Penurunan Suhu SA (di dalam tiap thread):** Proses SA pada suhu $T_2$ mutlak bergantung pada rute *current_solution* yang dihasilkan iterasi $T_1$ sebelumnya (*Markov Chain*). Ini tidak bisa diparalelkan secara internal.
4. **Reduksi Global Pemenang:** Komputasi mencari *thread* dengan rute absolut terpendek harus menunggu semua agen GPU selesai dan mengembalikan data ke CPU.

---

## 3. Mempartisi dan Memetakan Pekerjaan ke Pemroses [Menit 3:30 - 5:00]
**Bagian 2, Pertanyaan ke-1:** *"Bagaimana pekerjaan dibagi dan dipetakan pada versi MPI serta versi CUDA Anda? Tunjukkan hubungan antara indeks data global, rank MPI, indeks lokal, serta blockIdx dan threadIdx pada implementasi masing-masing."*

**Jawaban (Fokus CUDA):**
*(Diimplementasikan pada file: `src/vrp_sa.cu` baris ke-94)*

Pekerjaan dipartisi menggunakan eksekusi *Task Parallelism* melalui pemanfaatan struktur *Grid* satu dimensi (1D) yang menaungi *Blocks* dan *Threads*. Pemetaan didasarkan sepenuhnya pada *total thread* (agen pencari) yang diinstruksikan oleh pengguna.

Hubungan penentuan **ID/indeks global unik** dari setiap agen pencari SA diturunkan dari struktur Streaming Multiprocessor (SM) CUDA dengan rumusan linier:
```c
int id = threadIdx.x + blockIdx.x * blockDim.x;
```
- `threadIdx.x`: Indeks lokal thread di dalam sebuah Block $(0$ hingga $blockDim.x - 1)$.
- `blockIdx.x`: Koordinat/Indeks Block tersebut di dalam cakupan Grid.
- `blockDim.x`: Ukuran blok (total thread per block).

Dengan `id` global ini, setiap agen mengetahui persis memori mana miliknya:
1. Akses spesifik ke status *seed* generator acaknya sendiri (*cuRAND state*) dari `state[id]`.
2. Slot penulisan memori persis (tanpa tabrakan) untuk menyimpan skor hasil akhirnya: `d_best_solutions[id]`.

---

## 4. Distribusi Input, Output, dan Data Antara [Menit 5:00 - 6:30]
**Bagian 3, Pertanyaan ke-4:** *"Pada implementasi CUDA, data mana yang perlu ditransfer dari host ke device, data mana yang dapat tetap berada di device, dan data mana yang perlu dikembalikan ke host? Jelaskan juga alasan penggunaan global memory, shared memory, atau constant memory apabila digunakan."*

**Jawaban:**
*(Diimplementasikan pada file: `src/vrp_sa.cu` baris ke-174 untuk transfer H2D)*

1. **Ditransfer dari Host ke Device (H2D):**
   - Array `Node` (titik koordinat pelanggan, *demand* muatan, dan depot) disalin via `cudaMemcpyHostToDevice` karena pembacaan file aslinya hanya berhak dilakukan oleh CPU.
2. **Tetap berada di Device (Diciptakan dan musnah di Device):**
   - Array status (*state*) dari *random number generator* (`cuRAND state`). Ini dialokasikan di VRAM (`cudaMalloc`) dan dimodifikasi langsung oleh *kernel setup*. Tidak perlu ditarik kembali ke CPU.
   - Variabel dan *struct* rute sementara (seperti `current_sol`, `next_sol`) di dalam *kernel loop*. Ini dialokasikan otomatis oleh kompiler ke memori **Register / Local Memory** setiap *thread* sebagai ruang cakaran (*scratchpad*) privat yang berumur sangat pendek namun berkecepatan paling tinggi.
3. **Dikembalikan ke Host (D2H):**
   - Array raksasa rute terbaik (`d_best_solutions`) ditarik kembali via `cudaMemcpyDeviceToHost` agar CPU dapat melakukan komputasi pencarian 1 rute juara (reduksi mutlak).

*Pemilihan Global Memory:* Array `nodes` ditaruh di dalam **Global Memory** karena kapasitas datanya besar. Meskipun Global Memory secara teknis lambat, karena *array* koordinat ini bersifat *read-only* saat eksekusi (peta tidak berubah ukurannya), struktur *cache* L1/L2 GPU otomatis akan menangani serbuan akses bersama ini (*broadcasting cache*) dengan sangat efisien.

---

## 5. Mengoordinasikan Akses Data untuk Menghindari Konflik [Menit 6:30 - 8:30]
**Bagian 4, Pertanyaan ke-4:** *"Apabila banyak pekerja berkontribusi terhadap satu hasil, seperti jumlah total, histogram, atau nilai maksimum, bagaimana konflik pembaruan dicegah? Bandingkan pilihan akumulasi privat, reduksi bertahap, atau operasi atomik berdasarkan kebenaran, ketelitian, dan overhead."*

**Jawaban:**
*(Diimplementasikan pada file: `src/vrp_sa.cu` baris ke-139 untuk akumulasi privat, dan baris ke-218 untuk reduksi CPU)*

Banyak agen/pekerja menemukan solusi akhirnya dan berebut kontribusi menuju pencarian **Rute Minimum Global**. Jika puluhan ribu thread mencoba menimpa satu buah variabel tunggal `GlobalBest` saat menemukan rute bagus secara bersamaan, akan timbul tabrakan penulisan yang korup *(race condition)*.

Program kita menghindari konflik melalui teknik pendekatan **Akumulasi Privat (Private Output Array)** tanpa adanya persilangan area tulis *(write access)* sama sekali.
Setiap agen (thread) dipersilakan menulis hasilnya murni pada slot memori *(offset)* miliknya sendiri melalui kode: `d_best_solutions[id]`. Setelah seluruh balapan usai, penyortiran reduksi untuk mencari rute terkecil dikerjakan secara eksklusif oleh Host (CPU) pada tahap akhir menggunakan loop yang linear dan aman.

Perbandingan Teknik:
- **Operasi Atomik (Atomic Min):** Bawaan GPU menjamin 100% bebas konflik. Masalahnya, operasi atomik sangat lambat *(overhead yang mematikan kinerja)* jika banyak thread disuruh mengantre masuk menimpa satu alamat spesifik. Lagipula, tipe rute kita adalah *struct* kompleks dengan bertipe desimal (*double* jarak), sangat mustahil diatomasikan secara ringan.
- **Reduksi Bertahap (*Shared Memory Tree-Reduction*):** Menawarkan kinerja super cepat di GPU. Namun, kompleksitas logika kodenya untuk menciutkan ukuran *struct array* antar utas di *shared memory* lokal blok tergolong ekstrem.
- **Akumulasi Privat lalu Reduksi di Host (Sesuai Program):** Dijamin benar tanpa cacat (*thread-safe* mutlak). Kendati mengorbankan ukuran VRAM secara linier $O(N)$ mengikuti jumlah thread, pendekatan ini memberikan kemudahan desain paling kokoh (*safest trade-off*) agar GPU dapat menghabiskan waktu iterasi secara maksimal tanpa hambatan antrean memori.

---

## 6. Menjamin Urutan Pekerjaan Melalui Sinkronisasi [Menit 8:30 - 10:00]
**Bagian 5, Pertanyaan ke-5:** *"Bagaimana Anda menjamin urutan transfer host-to-device, eksekusi kernel, transfer device-to-host, dan penggunaan hasil oleh CPU? Tunjukkan dependensi yang digunakan pada satu stream maupun beberapa stream, termasuk peran event atau penantian host apabila diperlukan."*

**Jawaban:**
*(Diimplementasikan pada file: `src/main.cu` baris ke-54)*

Aliran ketertiban *(pipeline dependency)* program CUDA ini dijamin melalui barikade sinkronisasi yang tegas serta pemanfaatan tabiat eksekusi linear dari *Stream 0 (Default Stream)*.

1. **Host-to-Device (H2D) $\rightarrow$ Kernel Execution:** 
   Pemanggilan `cudaMemcpy(..., cudaMemcpyHostToDevice)` bersifat **Synchronous** yang membekukan status *Host*. Artinya, CPU terlarang mengalir memanggil *kernel* GPU jika arus penyalinan data belum genap 100% terparkir di VRAM.
2. **Kernel Execution $\rightarrow$ Device-to-Host (D2H):**
   Peluncuran kernel `run_simulated_annealing_kernel<<<...>>>` dikerjakan secara *Asynchronous* (CPU langsung meloncat membaca baris sintaks di bawahnya tanpa menunggu GPU). Guna mencegah kelalaian di mana CPU nekat mengekstrak data sementara GPU masih sibuk mutasi genetik rute SA, diletakkan palang peringatan mutlak `cudaDeviceSynchronize()`. Palang ini membekukan kaki tangan *Host* CPU sampai gema status *all-kernel-done* dari antrean GPU terdengar, baru setelah itu operasi *Memcpy* ditarik balik (D2H).
3. **Penyelarasan Presisi Waktu via Event Marker:**
   Untuk mengetahui kecepatan sejati GPU, dimasukkan dependensi alat ukur hardware `cudaEventRecord`. Tanda cetak waktu "Start" dan "Stop" disusun mengapit peluncuran *kernel*. CPU wajib ditahan menggunakan `cudaEventSynchronize(stop)` sebelum mengekstrak selisih milidetik waktu. Apabila tidak diblok, CPU bisa mencetak angka dini yang sangat kecil dan menyesatkan karena deretan sinyal di lintasan *Stream* belum selesai terbaca penuh oleh inti pemroses keras GPU.
