/**
 * @file vrp_sa.c
 * @brief Implementasi logika algoritma Simulated Annealing (SA), evaluasi solusi CVRP
 *        dengan batasan kapasitas armada, dan mutasi ketetanggaan (*neighborhood moves*).
 */

#include "vrp.h"

/**
 * @brief Mengevaluasi kelayakan rute tour dan menghitung akumulasi total jarak tempuh.
 *
 * Logika evaluasi (Capacitated VRP):
 * - Membaca urutan pelanggan secara sekuensial dari tour.
 * - Akumulasi muatan pelanggan ke kendaraan saat ini.
 * - Jika penambahan pelanggan berikutnya melampaui kapasitas Q, kendaraan saat ini
 *   kembali ke depot, lalu kendaraan berikutnya memulai rute dari depot.
 * - Total jarak adalah penjumlahan seluruh perjalanan pulang-pergi antar titik dan depot.
 *
 * @param sol Pointer ke Solusi yang akan dievaluasi.
 * @param nodes Array seluruh Node yang terdefinisi.
 * @param num_customers Jumlah total pelanggan.
 * @param capacity Kapasitas muatan maksimal per kendaraan.
 * @param out_distance Pointer output untuk total jarak tempuh yang dihitung.
 * @param out_vehicles Pointer output untuk jumlah kendaraan yang dikerahkan.
 */
void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity,
                       double *out_distance, int *out_vehicles) {
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

        /* Jika muatan saat ini ditambah permintaan pelanggan melebihi kapasitas */
        if (current_load + customer_node.demand > capacity && current_load > 0) {
            /* Kendaraan kembali ke depot */
            total_dist += calculate_distance(prev_node, depot);

            /* Kendaraan baru diberangkatkan dari depot */
            vehicles_count++;
            prev_node = depot;
            current_load = 0;
        }

        /* Bergerak menuju pelanggan dan memuat permintaannya */
        total_dist += calculate_distance(prev_node, customer_node);
        current_load += customer_node.demand;
        prev_node = customer_node;
    }

    /* Kendaraan terakhir kembali ke depot */
    total_dist += calculate_distance(prev_node, depot);

    *out_distance = total_dist;
    *out_vehicles = vehicles_count;
}

/**
 * @brief Menyalin isi struktur solusi dari src ke dest secara mendalam.
 *
 * @param dest Solusi target tujuan.
 * @param src Solusi sumber.
 * @param num_customers Jumlah pelanggan.
 */
void copy_solution(Solution *dest, const Solution *src, int num_customers) {
    memcpy(dest->tour, src->tour, sizeof(int) * num_customers);
    dest->total_distance = src->total_distance;
    dest->num_vehicles = src->num_vehicles;
}

/**
 * @brief Membentuk solusi awal dengan susunan pelanggan yang diacak (Fisher-Yates Shuffle).
 *
 * @param sol Solusi yang akan diinisialisasi.
 * @param num_customers Jumlah pelanggan (1..N).
 */
void generate_initial_solution(Solution *sol, int num_customers) {
    for (int i = 0; i < num_customers; i++) {
        sol->tour[i] = i + 1; /* ID pelanggan bernilai 1 hingga N */
    }

    /* Algoritma Fisher-Yates Shuffle untuk pengacakan merata */
    for (int i = num_customers - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = sol->tour[i];
        sol->tour[i] = sol->tour[j];
        sol->tour[j] = temp;
    }
}

/**
 * @brief Menerapkan modifikasi acak pada urutan rute (Neighborhood Move).
 *
 * Menggunakan dua jenis operator mutasi:
 * 1. Operator Swap: Menukar posisi dua pelanggan pada rute.
 * 2. Operator 2-Opt / Inversion: Membalik urutan segmen sub-rute antara dua titik acak.
 *
 * @param sol Pointer ke solusi yang dimodifikasi.
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
        /* Swap Operator */
        int temp = sol->tour[idx1];
        sol->tour[idx1] = sol->tour[idx2];
        sol->tour[idx2] = temp;
    } else {
        /* 2-Opt / Inversion Operator */
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
 * @brief Menjalankan siklus pendinginan Simulated Annealing secara penuh.
 *
 * Algoritma:
 * - Mulai dari suhu awal (T0).
 * - Di setiap suhu, lakukan eksplorasi ketetanggaan sejumlah iterations_per_temp.
 * - Solusi baru diterima jika lebih baik (Delta < 0) atau dengan probabilitas Boltzmann:
 *     P = exp(-Delta / T)
 * - Suhu diturunkan secara bertahap: T = T * cooling_rate.
 * - Berhenti saat suhu mencapai batas minimum (Tmin).
 *
 * @param sol Solusi awal sekaligus penampung solusi terbaik yang ditemukan.
 * @param nodes Array data seluruh Node.
 * @param num_customers Jumlah pelanggan.
 * @param capacity Kapasitas kendaraan.
 * @param params Parameter hiperparameter SA.
 */
void run_simulated_annealing(Solution *sol, const Node *nodes, int num_customers, int capacity,
                             const SAParameters *params) {
    Solution current_sol, neighbor_sol, best_sol;
    current_sol.tour = (int *)malloc(sizeof(int) * num_customers);
    neighbor_sol.tour = (int *)malloc(sizeof(int) * num_customers);
    best_sol.tour = (int *)malloc(sizeof(int) * num_customers);

    /* Inisialisasi solusi awal */
    copy_solution(&current_sol, sol, num_customers);
    evaluate_solution(&current_sol, nodes, num_customers, capacity,
                      &current_sol.total_distance, &current_sol.num_vehicles);
    copy_solution(&best_sol, &current_sol, num_customers);

    double temp = params->initial_temp;

    /* Perulangan penurunan suhu */
    while (temp > params->min_temp) {
        for (int iter = 0; iter < params->iterations_per_temp; iter++) {
            copy_solution(&neighbor_sol, &current_sol, num_customers);
            apply_neighborhood_move(&neighbor_sol, num_customers);
            evaluate_solution(&neighbor_sol, nodes, num_customers, capacity,
                              &neighbor_sol.total_distance, &neighbor_sol.num_vehicles);

            double delta = neighbor_sol.total_distance - current_sol.total_distance;

            /* Kriteria Penerimaan Metropolis-Boltzmann */
            if (delta < 0.0) {
                /* Solusi baru lebih pendek -> Selalu diterima */
                copy_solution(&current_sol, &neighbor_sol, num_customers);

                /* Simpan rekor baru solusi terbaik */
                if (current_sol.total_distance < best_sol.total_distance) {
                    copy_solution(&best_sol, &current_sol, num_customers);
                }
            } else {
                /* Solusi baru lebih panjang -> Diterima dengan probabilitas Boltzmann */
                double acceptance_prob = exp(-delta / temp);
                double random_val = (double)rand() / (double)RAND_MAX;

                if (random_val < acceptance_prob) {
                    copy_solution(&current_sol, &neighbor_sol, num_customers);
                }
            }
        }
        /* Penurunan suhu secara geometrik */
        temp *= params->cooling_rate;
    }

    /* Kembalikan solusi terbaik yang dicapai */
    copy_solution(sol, &best_sol, num_customers);

    /* Pembersihan memori internal */
    free(current_sol.tour);
    free(neighbor_sol.tour);
    free(best_sol.tour);
}
