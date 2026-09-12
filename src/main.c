/**
 * @file main.c
 * @brief Titik masuk utama (Entry Point) eksekusi paralel CVRP menggunakan MPI.
 *
 * Mengoordinasikan 4 fase eksekusi:
 * 1. Fase Inisialisasi & Broadcast Data (Master Node - Rank 0)
 * 2. Fase Eksekusi Paralel SA (Setiap Rank secara independen)
 * 3. Fase Reduksi Hasil (MPI_Reduce dengan MPI_MINLOC)
 * 4. Fase Rekonstruksi & Pelaporan Rute Terbaik (Master Node)
 */

#include "vrp.h"

int main(int argc, char **argv) {
    int rank, size;

    /* 1. Inisialisasi Lingkungan MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Konfigurasi Nilai Parameter Default */
    const char *dataset_filename = "data/dataset_small.txt";
    SAParameters sa_params;
    sa_params.initial_temp = 1000.0;
    sa_params.cooling_rate = 0.995;
    sa_params.min_temp = 0.001;
    sa_params.iterations_per_temp = 150;

    /* Membaca argumen baris perintah (CLI) jika dispesifikasikan pengguna:
     *   Arg 1: Path dataset
     *   Arg 2: Suhu awal (T0)
     *   Arg 3: Laju pendinginan (alpha)
     *   Arg 4: Suhu minimum (Tmin)
     *   Arg 5: Jumlah iterasi per suhu
     */
    if (argc > 1) dataset_filename = argv[1];
    if (argc > 2) sa_params.initial_temp = atof(argv[2]);
    if (argc > 3) sa_params.cooling_rate = atof(argv[3]);
    if (argc > 4) sa_params.min_temp = atof(argv[4]);
    if (argc > 5) sa_params.iterations_per_temp = atoi(argv[5]);

    int num_nodes = 0;
    int capacity = 0;
    Node *nodes = NULL;

    double start_time = 0.0;
    double end_time = 0.0;

    /* ===================================================================== */
    /* FASE 1: INISIALISASI & BROADCAST DATA (MASTER NODE - RANK 0)          */
    /* ===================================================================== */
    if (rank == 0) {
        printf("\n==================================================================\n");
        printf("   PARALLEL SIMULATED ANNEALING UNTUK OPTIMASI CVRP DENGAN MPI    \n");
        printf("==================================================================\n");
        printf("Jumlah Prosesor (Rank) : %d\n", size);
        printf("Dataset                : %s\n", dataset_filename);
        printf("Parameter SA           : T0=%.2f, Alpha=%.4f, Tmin=%.4f, Iters/Temp=%d\n",
               sa_params.initial_temp, sa_params.cooling_rate,
               sa_params.min_temp, sa_params.iterations_per_temp);

        if (read_dataset(dataset_filename, &nodes, &num_nodes, &capacity) != 0) {
            fprintf(stderr, "[Rank 0] Gagal memuat dataset '%s'. Eksekusi dihentikan.\n", dataset_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Dataset Berhasil Dimuat: %d Node (1 Depot, %d Pelanggan), Kapasitas=%d\n",
               num_nodes, num_nodes - 1, capacity);
        printf("------------------------------------------------------------------\n");
    }

    /* Penyelarasan waktu mulai eksekusi pada seluruh node */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    /* Broadcast metadata dasar: jumlah node dan kapasitas muatan kendaraan */
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&capacity, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_customers = num_nodes - 1;

    /* Alokasi memori array Node pada rank worker (Rank > 0) */
    if (rank != 0) {
        nodes = (Node *)malloc(sizeof(Node) * num_nodes);
        if (!nodes) {
            fprintf(stderr, "[Rank %d] Gagal alokasi memori array Node.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast array Node ke seluruh rank */
    MPI_Bcast(nodes, num_nodes * sizeof(Node), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ===================================================================== */
    /* FASE 2: EKSEKUSI SIMULATED ANNEALING PARALEL (SEMUA RANK)            */
    /* ===================================================================== */

    /* Random Seed unik per Rank untuk memastikan eksplorasi independen */
    unsigned int rank_seed = (unsigned int)(time(NULL) ^ (rank * 7919) ^ (rank << 5));
    srand(rank_seed);

    /* Inisialisasi solusi lokal */
    Solution local_sol;
    local_sol.tour = (int *)malloc(sizeof(int) * num_customers);

    generate_initial_solution(&local_sol, num_customers);
    evaluate_solution(&local_sol, nodes, num_customers, capacity,
                      &local_sol.total_distance, &local_sol.num_vehicles);

    /* Eksekusi Simulated Annealing mandiri pada prosesor saat ini */
    run_simulated_annealing(&local_sol, nodes, num_customers, capacity, &sa_params);

    /* Tunggu seluruh prosesor menyelesaikan annealing */
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double total_elapsed_time = end_time - start_time;

    /* ===================================================================== */
    /* FASE 3: REDUKSI HASIL GLOBAL (MPI_MINLOC)                             */
    /* ===================================================================== */

    /* Struktur pasangan (jarak, rank) untuk penentuan solusi terbaik */
    struct {
        double distance;
        int rank;
    } local_result, global_result;

    local_result.distance = local_sol.total_distance;
    local_result.rank = rank;

    /* Master mencari jarak minimum dari seluruh rank */
    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    /* ===================================================================== */
    /* FASE 4: REKONSTRUKSI & OUTPUT HASIL OPTIMAL (MASTER NODE)             */
    /* ===================================================================== */

    if (rank == 0) {
        int winning_rank = global_result.rank;
        double min_global_distance = global_result.distance;

        Solution final_solution;
        final_solution.tour = (int *)malloc(sizeof(int) * num_customers);

        if (winning_rank == 0) {
            /* Solusi terbaik ditemukan oleh Master sendiri */
            copy_solution(&final_solution, &local_sol, num_customers);
        } else {
            /* Menerima array urutan rute dari Worker pemenang */
            MPI_Recv(final_solution.tour, num_customers, MPI_INT, winning_rank, 99,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            evaluate_solution(&final_solution, nodes, num_customers, capacity,
                              &final_solution.total_distance, &final_solution.num_vehicles);
        }

        /* Tampilkan rangkuman hasil optimasi ke layar */
        printf("\n==================================================================\n");
        printf("                     HASIL OPTIMASI GLOBAL                        \n");
        printf("==================================================================\n");
        printf("Rank Pemenang (Best Solution) : Rank %d\n", winning_rank);
        printf("Total Jarak Tempuh Minimum    : %.4f unit\n", min_global_distance);
        printf("Total Kendaraan yang Digunakan: %d armada\n", final_solution.num_vehicles);
        printf("Waktu Komputasi Paralel       : %.4f detik\n", total_elapsed_time);

        /* Cetak rincian navigasi per armada kendaraan */
        print_detailed_routes(&final_solution, nodes, num_customers, capacity);

        free(final_solution.tour);
    } else {
        /* Jika rank ini adalah pemenang (bukan rank 0), kirim susunan tour ke Master */
        if (rank == global_result.rank) {
            MPI_Send(local_sol.tour, num_customers, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }
    }

    /* Pembersihan alokasi memori */
    free(local_sol.tour);
    free(nodes);

    /* 5. Finalisasi Lingkungan MPI */
    MPI_Finalize();
    return 0;
}
