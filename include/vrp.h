/**
 * @file vrp.h
 * @brief Definisi struktur data, konstanta, dan prototipe fungsi untuk penyelesaian
 *        Capacitated Vehicle Routing Problem (CVRP) menggunakan Parallel Simulated Annealing.
 *
 * Header ini mengintegrasikan seluruh komponen modul:
 * - Struktur data Node, Solution, dan SAParameters
 * - Fungsi utilitas I/O dan perhitungan jarak (vrp_utils.c)
 * - Algoritma evaluasi rute, mutasi ketetanggaan, dan annealing (vrp_sa.c)
 */

#ifndef VRP_H
#define VRP_H

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
 * @brief Menyimpan data setiap titik lokasi pada grafik VRP (Depot atau Pelanggan).
 *
 * @var Node::id ID unik node (0 mewakili Depot, 1..N mewakili Pelanggan).
 * @var Node::x Koordinat Kartesius sumbu X.
 * @var Node::y Koordinat Kartesius sumbu Y.
 * @var Node::demand Besarnya permintaan muatan pelanggan (bernilai 0 untuk Depot).
 */
typedef struct {
    int id;
    double x;
    double y;
    int demand;
} Node;

/**
 * @struct Solution
 * @brief Representasi kandidat solusi CVRP berupa urutan permutasi kunjungan pelanggan.
 *
 * @var Solution::tour Array permutasi ID pelanggan yang dikunjungi (panjang = num_customers).
 * @var Solution::total_distance Total akumulasi jarak tempuh seluruh rute kendaraan.
 * @var Solution::num_vehicles Jumlah kendaraan yang dikerahkan untuk melayani seluruh rute.
 */
typedef struct {
    int *tour;
    double total_distance;
    int num_vehicles;
} Solution;

/**
 * @struct SAParameters
 * @brief Konfigurasi hiperparameter untuk algoritma Simulated Annealing.
 *
 * @var SAParameters::initial_temp Suhu awal sistem (T0).
 * @var SAParameters::cooling_rate Laju penurunan suhu geometrik (alpha), misalnya 0.995.
 * @var SAParameters::min_temp Suhu batas penghentian pendinginan (Tmin).
 * @var SAParameters::iterations_per_temp Jumlah percobaan ketetanggaan pada setiap suhu.
 */
typedef struct {
    double initial_temp;
    double cooling_rate;
    double min_temp;
    int iterations_per_temp;
} SAParameters;

/* ========================================================================= */
/*                   PROTOTIPE FUNGSI UTILITAS (vrp_utils.c)                 */
/* ========================================================================= */

/**
 * @brief Menghitung jarak Euclidean 2 dimensi antara dua Node.
 * @param a Node pertama.
 * @param b Node kedua.
 * @return Jarak garis lurus antara titik a dan b.
 */
double calculate_distance(Node a, Node b);

/**
 * @brief Membaca file dataset CVRP dan mem-parsing data koordinat serta kapasitas.
 * @param filename Lokasi path file dataset.
 * @param nodes_out Output pointer array Node yang dialokasikan secara dinamis.
 * @param num_nodes_out Output jumlah total node (Depot + Pelanggan).
 * @param capacity_out Output kapasitas muatan maksimum per kendaraan.
 * @return 0 jika sukses, -1 jika gagal membaca atau format tidak valid.
 */
int read_dataset(const char *filename, Node **nodes_out, int *num_nodes_out, int *capacity_out);

/**
 * @brief Menampilkan rute terperinci per kendaraan ke terminal.
 * @param sol Solusi CVRP yang akan dicetak.
 * @param nodes Array seluruh Node yang terdefinisi.
 * @param num_customers Jumlah pelanggan.
 * @param capacity Kapasitas maksimum per kendaraan.
 */
void print_detailed_routes(const Solution *sol, const Node *nodes, int num_customers, int capacity);

/* ========================================================================= */
/*               PROTOTIPE FUNGSI SIMULATED ANNEALING (vrp_sa.c)             */
/* ========================================================================= */

/**
 * @brief Mengevaluasi kelayakan rute tour dan menghitung total jarak tempuh
 *        berdasarkan batasan kapasitas kendaraan.
 * @param sol Solusi yang akan dievaluasi.
 * @param nodes Array seluruh Node.
 * @param num_customers Jumlah total pelanggan.
 * @param capacity Batas kapasitas kendaraan.
 * @param out_distance Pointer untuk menyimpan total jarak yang dihitung.
 * @param out_vehicles Pointer untuk menyimpan jumlah kendaraan yang digunakan.
 */
void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity,
                       double *out_distance, int *out_vehicles);

/**
 * @brief Melakukan penyalinan mendalam (deep copy) dari objek Solusi sumber ke tujuan.
 * @param dest Solusi tujuan.
 * @param src Solusi sumber.
 * @param num_customers Jumlah pelanggan dalam tour.
 */
void copy_solution(Solution *dest, const Solution *src, int num_customers);

/**
 * @brief Membentuk solusi awal dengan susunan acak pelanggan menggunakan Fisher-Yates shuffle.
 * @param sol Objek Solusi yang akan diinisialisasi.
 * @param num_customers Jumlah pelanggan.
 */
void generate_initial_solution(Solution *sol, int num_customers);

/**
 * @brief Menghasilkan variasi solusi baru dengan operator Swap atau 2-Opt/Inversion.
 * @param sol Objek Solusi yang dimodifikasi.
 * @param num_customers Jumlah pelanggan dalam rute.
 */
void apply_neighborhood_move(Solution *sol, int num_customers);

/**
 * @brief Menjalankan siklus algoritma Simulated Annealing secara penuh hingga sistem membeku.
 * @param sol Solusi awal yang sekaligus menjadi penampung solusi terbaik yang ditemukan.
 * @param nodes Data seluruh Node.
 * @param num_customers Jumlah pelanggan.
 * @param capacity Batas kapasitas kendaraan.
 * @param params Konfigurasi parameter SA.
 */
void run_simulated_annealing(Solution *sol, const Node *nodes, int num_customers, int capacity,
                             const SAParameters *params);

#endif /* VRP_H */
