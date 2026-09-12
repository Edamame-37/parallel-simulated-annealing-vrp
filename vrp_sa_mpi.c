/**
 * @file vrp_sa_mpi.c
 * @brief Implementasi Parallel Simulated Annealing untuk Capacitated Vehicle Routing Problem (CVRP) menggunakan MPI.
 *
 * Program ini mengimplementasikan algoritma metaheuristik Simulated Annealing (SA)
 * secara terdistribusi/paralel dengan pendekatan Multiple Independent Markov Chains.
 * Setiap rank (prosesor MPI) mengeksplorasi ruang solusi CVRP secara independen
 * dengan random seed yang unik. Di akhir proses, hasil optimal global direduksi
 * menggunakan MPI_Reduce (MPI_MINLOC) dan dikumpulkan oleh Master (Rank 0).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

/* ========================================================================= */
/*                          STRUKTUR DATA PROGRAM                            */
/* ========================================================================= */

/**
 * @struct Node
 * @brief Menyimpan data setiap titik lokasi (Depot atau Pelanggan).
 *
 * @var Node::id ID unik node (0 untuk Depot, 1..N untuk Pelanggan).
 * @var Node::x Koordinat sumbu X.
 * @var Node::y Koordinat sumbu Y.
 * @var Node::demand Permintaan muatan pelanggan (0 untuk Depot).
 */
typedef struct {
    int id;
    double x;
    double y;
    int demand;
} Node;

/**
 * @struct Solution
 * @brief Representasi solusi CVRP berupa urutan permutasi pelanggan (*giant tour*).
 *
 * @var Solution::tour Array urutan kunjungan seluruh pelanggan (panjang = num_customers).
 * @var Solution::total_distance Total akumulasi jarak tempuh seluruh armada kendaraan.
 * @var Solution::num_vehicles Jumlah kendaraan yang dibutuhkan sesuai batas kapasitas.
 */
typedef struct {
    int *tour;
    double total_distance;
    int num_vehicles;
} Solution;

/**
 * @struct SAParameters
 * @brief Konfigurasi parameter untuk algoritma Simulated Annealing.
 */
typedef struct {
    double initial_temp;      /**< Suhu awal (T0) */
    double cooling_rate;      /**< Laju pendinginan geometrik (alpha), misal 0.995 */
    double min_temp;          /**< Batas suhu minimum untuk berhenti (Tmin) */
    int iterations_per_temp;  /**< Jumlah iterasi pencarian ketetanggaan pada setiap tingkat suhu */
} SAParameters;

/* ========================================================================= */
/*                       DEKLARASI PROTOTIPE FUNGSI                          */
/* ========================================================================= */

double calculate_distance(Node a, Node b);
void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity, double *out_distance, int *out_vehicles);
void copy_solution(Solution *dest, const Solution *src, int num_customers);
void generate_initial_solution(Solution *sol, int num_customers);
void apply_neighborhood_move(Solution *sol, int num_customers);
int read_dataset(const char *filename, Node **nodes_out, int *num_nodes_out, int *capacity_out);
void print_detailed_routes(const Solution *sol, const Node *nodes, int num_customers, int capacity);

/* ========================================================================= */
/*                         IMPLEMENTASI FUNGSI HELPER                        */
/* ========================================================================= */

/**
 * @brief Menghitung jarak Euclidean antara dua titik (Node).
 *
 * Rumus: sqrt((x2 - x1)^2 + (y2 - y1)^2)
 *
 * @param a Node asal
 * @param b Node tujuan
 * @return Nilai jarak Euclidean dalam tipe double.
 */
double calculate_distance(Node a, Node b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt((dx * dx) + (dy * dy));
}

/**
 * @brief Membaca file dataset CVRP.
 *
 * Format berkas yang didukung:
 * Baris komentar diawali dengan '#' diabaikan.
 * Baris data pertama: JUMLAH_NODE KAPASITAS_KENDARAAN
 * Baris-baris berikutnya: ID X Y DEMAND
 *
 * @param filename Nama berkas dataset.
 * @param nodes_out Pointer ke array Node hasil alokasi.
 * @param num_nodes_out Pointer ke jumlah total node (Depot + Pelanggan).
 * @param capacity_out Pointer ke kapasitas maksimum per kendaraan.
 * @return 0 jika sukses, -1 jika gagal membuka atau membaca berkas.
 */
int read_dataset(const char *filename, Node **nodes_out, int *num_nodes_out, int *capacity_out) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] Gagal membuka file dataset: %s\n", filename);
        return -1;
    }

    char line[256];
    int num_nodes = 0;
    int capacity = 0;

    /* Membaca header untuk mendapatkan jumlah node dan kapasitas */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sscanf(line, "%d %d", &num_nodes, &capacity) == 2) {
            break;
        }
    }

    if (num_nodes <= 1 || capacity <= 0) {
        fprintf(stderr, "[ERROR] Format header dataset tidak valid (nodes: %d, capacity: %d)\n", num_nodes, capacity);
        fclose(fp);
        return -1;
    }

    /* Alokasi memori untuk seluruh node (Depot index 0, Pelanggan index 1..num_nodes-1) */
    Node *nodes = (Node *)malloc(sizeof(Node) * num_nodes);
    if (!nodes) {
        fprintf(stderr, "[ERROR] Gagal mengalokasikan memori untuk array Node.\n");
        fclose(fp);
        return -1;
    }

    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < num_nodes) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        int id;
        double x, y;
        int demand;
        if (sscanf(line, "%d %lf %lf %d", &id, &x, &y, &demand) == 4) {
            nodes[count].id = id;
            nodes[count].x = x;
            nodes[count].y = y;
            nodes[count].demand = demand;
            count++;
        }
    }

    fclose(fp);

    if (count != num_nodes) {
        fprintf(stderr, "[ERROR] Jumlah node yang terbaca (%d) tidak sesuai header (%d).\n", count, num_nodes);
        free(nodes);
        return -1;
    }

    *nodes_out = nodes;
    *num_nodes_out = num_nodes;
    *capacity_out = capacity;
    return 0;
}

/**
 * @brief Mengevaluasi rute CVRP berdasarkan batas kapasitas muatan kendaraan.
 *
 * Pelanggan dilayani secara berurutan sesuai urutan tour. Jika penambahan pelanggan berikutnya
 * akan melebihi kapasitas kendaraan, kendaraan saat ini kembali ke depot dan kendaraan baru
 * diberangkatkan dari depot.
 *
 * @param sol Pointer ke Solusi yang akan dievaluasi.
 * @param nodes Array data seluruh Node (index 0 adalah Depot).
 * @param num_customers Jumlah total pelanggan (num_nodes - 1).
 * @param capacity Kapasitas maksimal kendaraan.
 * @param out_distance Pointer output untuk total jarak tempuh.
 * @param out_vehicles Pointer output untuk jumlah kendaraan yang digunakan.
 */
void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity, double *out_distance, int *out_vehicles) {
    if (num_customers <= 0) {
        *out_distance = 0.0;
        *out_vehicles = 0;
        return;
    }

    double total_dist = 0.0;
    int vehicles_count = 1;
    int current_load = 0;

    Node depot = nodes[0];
    Node prev_node = depot;

    for (int i = 0; i < num_customers; i++) {
        int customer_id = sol->tour[i];
        Node customer_node = nodes[customer_id];

        /* Jika menambahkan pelanggan melebihi kapasitas, rute kendaraan saat ini selesai */
        if (current_load + customer_node.demand > capacity && current_load > 0) {
            /* Kembali dari pelanggan sebelumnya ke Depot */
            total_dist += calculate_distance(prev_node, depot);

            /* Kendaraan baru berangkat dari Depot */
            vehicles_count++;
            prev_node = depot;
            current_load = 0;
        }

        /* Perjalanan dari node sebelumnya ke pelanggan saat ini */
        total_dist += calculate_distance(prev_node, customer_node);
        current_load += customer_node.demand;
        prev_node = customer_node;
    }

    /* Kendaraan terakhir kembali ke Depot */
    total_dist += calculate_distance(prev_node, depot);

    *out_distance = total_dist;
    *out_vehicles = vehicles_count;
}

/**
 * @brief Menyalin isi struktur solusi dari src ke dest.
 *
 * @param dest Solusi target.
 * @param src Solusi sumber.
 * @param num_customers Jumlah pelanggan.
 */
void copy_solution(Solution *dest, const Solution *src, int num_customers) {
    memcpy(dest->tour, src->tour, sizeof(int) * num_customers);
    dest->total_distance = src->total_distance;
    dest->num_vehicles = src->num_vehicles;
}

/**
 * @brief Menghasilkan solusi awal secara acak (permutasi acak pelanggan 1..N).
 *
 * @param sol Solusi yang akan diinisialisasi.
 * @param num_customers Jumlah pelanggan.
 */
void generate_initial_solution(Solution *sol, int num_customers) {
    for (int i = 0; i < num_customers; i++) {
        sol->tour[i] = i + 1; /* ID pelanggan bernilai 1 hingga N */
    }

    /* Fisher-Yates Shuffle untuk mengacak urutan rute awal */
    for (int i = num_customers - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = sol->tour[i];
        sol->tour[i] = sol->tour[j];
        sol->tour[j] = temp;
    }
}

/**
 * @brief Menerapkan modifikasi ketetanggaan (Neighborhood Move).
 *
 * Mendukung kombinasi mutasi:
 * 1. Swap: Menukar posisi dua pelanggan acak.
 * 2. 2-Opt / Inversion: Membalik urutan sub-rute antara dua indeks acak.
 *
 * @param sol Pointer ke solusi yang akan dimodifikasi langsung.
 * @param num_customers Jumlah pelanggan.
 */
void apply_neighborhood_move(Solution *sol, int num_customers) {
    if (num_customers < 2) return;

    int idx1 = rand() % num_customers;
    int idx2 = rand() % num_customers;
    while (idx1 == idx2) {
        idx2 = rand() % num_customers;
    }

    int move_type = rand() % 2;

    if (move_type == 0) {
        /* Swap Operator: Menukar dua pelanggan */
        int temp = sol->tour[idx1];
        sol->tour[idx1] = sol->tour[idx2];
        sol->tour[idx2] = temp;
    } else {
        /* 2-Opt / Inversion Operator: Membalikkan rentang [min(idx1, idx2), max(idx1, idx2)] */
        int start = (idx1 < idx2) ? idx1 : idx2;
        int end = (idx1 > idx2) ? idx1 : idx2;

        while (start < end) {
            int temp = sol->tour[start];
            sol->tour[start] = sol->tour[end];
            sol->tour[end] = temp;
            start++;
            end--;
        }
    }
}

/**
 * @brief Mencetak visualisasi detail rute perjalanan per kendaraan.
 *
 * @param sol Solusi optimal yang akan dicetak.
 * @param nodes Array data seluruh Node.
 * @param num_customers Jumlah total pelanggan.
 * @param capacity Kapasitas maksimal kendaraan.
 */
void print_detailed_routes(const Solution *sol, const Node *nodes, int num_customers, int capacity) {
    printf("\n==================================================================\n");
    printf("                  RINCIAN RUTE KENDARAAN (CVRP)                  \n");
    printf("==================================================================\n");

    int vehicle_num = 1;
    int current_load = 0;
    double route_dist = 0.0;
    Node depot = nodes[0];
    Node prev = depot;

    printf("[Kendaraan #%d]\n  Rute: Depot (ID: 0)", vehicle_num);

    for (int i = 0; i < num_customers; i++) {
        int cust_id = sol->tour[i];
        Node curr = nodes[cust_id];

        if (current_load + curr.demand > capacity && current_load > 0) {
            route_dist += calculate_distance(prev, depot);
            printf(" -> Depot (ID: 0)\n");
            printf("  Muatan: %d / %d | Sub-Jarak: %.2f\n\n", current_load, capacity, route_dist);

            vehicle_num++;
            current_load = 0;
            route_dist = 0.0;
            prev = depot;
            printf("[Kendaraan #%d]\n  Rute: Depot (ID: 0)", vehicle_num);
        }

        route_dist += calculate_distance(prev, curr);
        current_load += curr.demand;
        printf(" -> P%d(D:%d)", curr.id, curr.demand);
        prev = curr;
    }

    route_dist += calculate_distance(prev, depot);
    printf(" -> Depot (ID: 0)\n");
    printf("  Muatan: %d / %d | Sub-Jarak: %.2f\n", current_load, capacity, route_dist);
    printf("==================================================================\n\n");
}

/* ========================================================================= */
/*                             PROGRAM UTAMA (MAIN)                          */
/* ========================================================================= */

int main(int argc, char **argv) {
    int rank, size;

    /* 1. Inisialisasi Lingkungan MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Konfigurasi Parameter Default */
    const char *dataset_filename = "data/dataset_small.txt";
    SAParameters sa_params;
    sa_params.initial_temp = 1000.0;
    sa_params.cooling_rate = 0.995;
    sa_params.min_temp = 0.001;
    sa_params.iterations_per_temp = 150;

    /* Parsing argumen baris perintah (CLI) jika diberikan */
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
    /* FASE 1: INISIALISASI DAN DISTRIBUSI DATA (MASTER NODE - RANK 0)       */
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
            fprintf(stderr, "[Rank 0] Gagal memuat dataset. Menghentikan program.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Dataset Berhasil Dimuat: %d Node (1 Depot, %d Pelanggan), Kapasitas=%d\n",
               num_nodes, num_nodes - 1, capacity);
        printf("------------------------------------------------------------------\n");
    }

    /* Mulai pencatatan waktu eksekusi serentak */
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    /* Broadcast metadata dasar: jumlah node dan kapasitas kendaraan */
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&capacity, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_customers = num_nodes - 1;

    /* Alokasi array nodes pada Worker nodes (Rank > 0) */
    if (rank != 0) {
        nodes = (Node *)malloc(sizeof(Node) * num_nodes);
        if (!nodes) {
            fprintf(stderr, "[Rank %d] Gagal alokasi memori nodes.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast seluruh data node ke seluruh worker */
    /* Menggunakan MPI_BYTE karena struct Node bertipe plain-old-data (POD) */
    MPI_Bcast(nodes, num_nodes * sizeof(Node), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ===================================================================== */
    /* FASE 2: EKSEKUSI SIMULATED ANNEALING PARALEL (SEMUA RANK)            */
    /* ===================================================================== */

    /* Random Seed unik di setiap Rank untuk memastikan eksplorasi independen */
    unsigned int rank_seed = (unsigned int)(time(NULL) ^ (rank * 7919) ^ (rank << 5));
    srand(rank_seed);

    /* Alokasi memori untuk solusi lokal */
    Solution current_sol, neighbor_sol, best_sol;
    current_sol.tour = (int *)malloc(sizeof(int) * num_customers);
    neighbor_sol.tour = (int *)malloc(sizeof(int) * num_customers);
    best_sol.tour = (int *)malloc(sizeof(int) * num_customers);

    /* Pembentukan solusi awal secara acak */
    generate_initial_solution(&current_sol, num_customers);
    evaluate_solution(&current_sol, nodes, num_customers, capacity,
                      &current_sol.total_distance, &current_sol.num_vehicles);

    /* Inisialisasi solusi terbaik lokal dengan solusi awal */
    copy_solution(&best_sol, &current_sol, num_customers);

    double temp = sa_params.initial_temp;

    /* Loop Pendinginan Simulated Annealing */
    while (temp > sa_params.min_temp) {
        for (int iter = 0; iter < sa_params.iterations_per_temp; iter++) {
            /* Bentuk kandidat tetangga dari solusi saat ini */
            copy_solution(&neighbor_sol, &current_sol, num_customers);
            apply_neighborhood_move(&neighbor_sol, num_customers);
            evaluate_solution(&neighbor_sol, nodes, num_customers, capacity,
                              &neighbor_sol.total_distance, &neighbor_sol.num_vehicles);

            double delta = neighbor_sol.total_distance - current_sol.total_distance;

            /* Kriteria Penerimaan Metropolis-Boltzmann */
            if (delta < 0.0) {
                /* Solusi baru lebih baik -> Selalu diterima */
                copy_solution(&current_sol, &neighbor_sol, num_customers);

                /* Perbarui solusi terbaik lokal jika mencetak rekor baru */
                if (current_sol.total_distance < best_sol.total_distance) {
                    copy_solution(&best_sol, &current_sol, num_customers);
                }
            } else {
                /* Solusi baru lebih buruk -> Diterima dengan probabilitas e^(-delta / temp) */
                double acceptance_prob = exp(-delta / temp);
                double random_val = (double)rand() / (double)RAND_MAX;

                if (random_val < acceptance_prob) {
                    copy_solution(&current_sol, &neighbor_sol, num_customers);
                }
            }
        }
        /* Penurunan suhu secara geometrik */
        temp *= sa_params.cooling_rate;
    }

    /* Sinkronisasi sebelum reduksi hasil */
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double local_execution_time = end_time - start_time;

    /* ===================================================================== */
    /* FASE 3: REDUKSI DAN PENENTUAN JUARA GLOBAL (MPI_MINLOC)               */
    /* ===================================================================== */

    /* Struktur data untuk MPI_DOUBLE_INT pada reduksi nilai minimum beserta index rank */
    struct {
        double distance;
        int rank;
    } local_result, global_result;

    local_result.distance = best_sol.total_distance;
    local_result.rank = rank;

    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    /* ===================================================================== */
    /* FASE 4: REKONSTRUKSI DAN OUTPUT HASIL OPTIMAL (MASTER NODE)           */
    /* ===================================================================== */

    if (rank == 0) {
        int winning_rank = global_result.rank;
        double min_global_distance = global_result.distance;

        Solution final_best_solution;
        final_best_solution.tour = (int *)malloc(sizeof(int) * num_customers);

        if (winning_rank == 0) {
            /* Solusi terbaik ditemukan oleh Master sendiri */
            copy_solution(&final_best_solution, &best_sol, num_customers);
        } else {
            /* Menerima array tour dari Worker yang memenangkan jarak minimum */
            MPI_Recv(final_best_solution.tour, num_customers, MPI_INT, winning_rank, 99,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            evaluate_solution(&final_best_solution, nodes, num_customers, capacity,
                              &final_best_solution.total_distance, &final_best_solution.num_vehicles);
        }

        /* Cetak ringkasan hasil optimasi global */
        printf("\n==================================================================\n");
        printf("                     HASIL OPTIMASI GLOBAL                        \n");
        printf("==================================================================\n");
        printf("Rank Pemenang (Best Solution) : Rank %d\n", winning_rank);
        printf("Total Jarak Tempuh Minimum    : %.4f unit\n", min_global_distance);
        printf("Total Kendaraan yang Digunakan: %d armada\n", final_best_solution.num_vehicles);
        printf("Waktu Komputasi Paralel       : %.4f detik\n", local_execution_time);

        /* Cetak rute detail */
        print_detailed_routes(&final_best_solution, nodes, num_customers, capacity);

        free(final_best_solution.tour);
    } else {
        /* Jika rank ini adalah pemenang (dan bukan rank 0), kirimkan array tour ke Master */
        if (rank == global_result.rank) {
            MPI_Send(best_sol.tour, num_customers, MPI_INT, 0, 99, MPI_COMM_WORLD);
        }
    }

    /* Pembersihan memori */
    free(current_sol.tour);
    free(neighbor_sol.tour);
    free(best_sol.tour);
    free(nodes);

    /* Finalisasi MPI */
    MPI_Finalize();
    return 0;
}
