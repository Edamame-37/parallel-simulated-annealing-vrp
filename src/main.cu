/**
 * @file main.cu
 * @brief Titik masuk utama (Entry Point) eksekusi paralel CVRP menggunakan GPU CUDA.
 *
 * Mengoordinasikan 4 fase eksekusi:
 * 1. Fase Inisialisasi & Memuat Dataset (CPU Host)
 * 2. Fase Konfigurasi Grid/Block CUDA & Parameter SA
 * 3. Fase Eksekusi Massively Parallel SA pada GPU (Panggilan Wrapper run_cuda_parallel_sa)
 * 4. Fase Pelaporan Rute & Evaluasi Performa
 */

#include "vrp.h"

int main(int argc, char **argv) {
    /* Konfigurasi Nilai Parameter Default SA */
    const char *dataset_filename = "data/dataset_small.txt";
    SAParameters sa_params;
    sa_params.initial_temp = 1000.0;
    sa_params.cooling_rate = 0.995;
    sa_params.min_temp = 0.001;
    sa_params.iterations_per_temp = 150;

    /* Konfigurasi Default GPU CUDA Grid & Block */
    int num_blocks = 16;           /* Jumlah CUDA Block */
    int threads_per_block = 128;    /* Jumlah Thread per Block */

    /* Membaca argumen baris perintah (CLI) jika dispesifikasikan pengguna:
     *   Arg 1: Path dataset
     *   Arg 2: Suhu awal (T0)
     *   Arg 3: Laju pendinginan (alpha)
     *   Arg 4: Suhu minimum (Tmin)
     *   Arg 5: Jumlah iterasi per suhu
     *   Arg 6: Jumlah CUDA Blocks
     *   Arg 7: Jumlah Threads per Block
     */
    if (argc > 1) dataset_filename = argv[1];
    if (argc > 2) sa_params.initial_temp = atof(argv[2]);
    if (argc > 3) sa_params.cooling_rate = atof(argv[3]);
    if (argc > 4) sa_params.min_temp = atof(argv[4]);
    if (argc > 5) sa_params.iterations_per_temp = atoi(argv[5]);
    if (argc > 6) num_blocks = atoi(argv[6]);
    if (argc > 7) threads_per_block = atoi(argv[7]);

    int total_gpu_threads = num_blocks * threads_per_block;

    /* Cetak Header Program */
    printf("\n==================================================================\n");
    printf("   MASSIVELY PARALLEL SIMULATED ANNEALING CVRP DENGAN GPU CUDA    \n");
    printf("==================================================================\n");
    printf("Dataset               : %s\n", dataset_filename);
    printf("Konfigurasi GPU       : %d Blocks x %d Threads/Block (%d Total Threads)\n",
           num_blocks, threads_per_block, total_gpu_threads);
    printf("Parameter SA          : T0=%.2f, Alpha=%.4f, Tmin=%.4f, Iters/Temp=%d\n",
           sa_params.initial_temp, sa_params.cooling_rate,
           sa_params.min_temp, sa_params.iterations_per_temp);

    /* ===================================================================== */
    /* FASE 1: MEMUAT DATASET PADA HOST (CPU)                                */
    /* ===================================================================== */
    int num_nodes = 0;
    int capacity = 0;
    Node *nodes = NULL;

    if (read_dataset(dataset_filename, &nodes, &num_nodes, &capacity) != 0) {
        fprintf(stderr, "[ERROR] Gagal memuat dataset '%s'. Eksekusi dihentikan.\n", dataset_filename);
        return EXIT_FAILURE;
    }

    int num_customers = num_nodes - 1;

    /* Validasi batas maksimum pelanggan pada struct Solution CUDA */
    if (num_customers > MAX_CUSTOMERS) {
        fprintf(stderr, "[ERROR] Jumlah pelanggan (%d) melebihi MAX_CUSTOMERS (%d).\n", 
                num_customers, MAX_CUSTOMERS);
        free(nodes);
        return EXIT_FAILURE;
    }

    printf("Dataset Berhasil Dimuat: %d Node (1 Depot, %d Pelanggan), Kapasitas=%d\n",
           num_nodes, num_customers, capacity);
    printf("------------------------------------------------------------------\n");

    /* ===================================================================== */
    /* FASE 2 & 3: EKSEKUSI CUDA PARALLEL SA & PENGUKURAN WAKTU              */
    /* ===================================================================== */
    Solution global_best_sol;
    
    /* Pengukuran Waktu Presisi Tinggi Menggunakan CUDA Events */
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, 0));

    /* Memanggil Host Wrapper untuk Menjalankan Kernel CUDA */
    run_cuda_parallel_sa(nodes, num_customers, capacity, sa_params,
                        num_blocks, threads_per_block, &global_best_sol);

    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    double total_elapsed_time = elapsed_ms / 1000.0; /* Konversi milidetik ke detik */

    /* ===================================================================== */
    /* FASE 4: PELAPORAN HASIL OPTIMASI GLOBAL                               */
    /* ===================================================================== */
    printf("\n==================================================================\n");
    printf("                     HASIL OPTIMASI GLOBAL (GPU)                  \n");
    printf("==================================================================\n");
    printf("Total Jarak Tempuh Minimum : %.4f unit\n", global_best_sol.total_distance);
    printf("Total Kendaraan Digunakan  : %d armada\n", global_best_sol.num_vehicles);
    printf("Waktu Komputasi GPU CUDA   : %.4f detik (%.2f ms)\n", 
           total_elapsed_time, elapsed_ms);

    /* Cetak rincian rute per armada kendaraan */
    print_detailed_routes(&global_best_sol, nodes, num_customers, capacity);

    /* Pembersihan Memori & Resource CUDA */
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    free(nodes);

    printf("\n[INFO] Eksekusi GPU CUDA Selesai dengan Sukses.\n");
    return EXIT_SUCCESS;
}