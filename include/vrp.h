/**
 * @file vrp.h
 * @brief Definisi struktur data, konstanta, dan prototipe fungsi untuk penyelesaian
 *        Capacitated Vehicle Routing Problem (CVRP) menggunakan Massively Parallel 
 *        Simulated Annealing pada GPU CUDA.
 *
 * Header ini mengintegrasikan seluruh modul CUDA:
 * - Struktur data Node, Solution, dan SAParameters dengan CUDA Qualifiers
 * - Pustaka CUDA Runtime & cuRAND untuk generasi angka acak paralel
 * - Fungsi utilitas I/O Host (CPU) dan Device (GPU)
 * - Kernel CUDA (__global__) untuk eksekusi Simulated Annealing secara massal
 */

#ifndef VRP_H
#define VRP_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Header Khusus CUDA */
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* ========================================================================= */
/*                               KONSTANTA PROGRAM                           */
/* ========================================================================= */

/** 
 * Batas maksimum pelanggan untuk alokasi memori lokal/register pada thread GPU.
 * Menghindari alokasi memori dinamis (malloc) di dalam kernel CUDA demi performa tinggi.
 */
#define MAX_CUSTOMERS 256

/* Macro penanganan error CUDA */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* ========================================================================= */
/*                          STRUKTUR DATA PROGRAM                            */
/* ========================================================================= */

/**
 * @struct Node
 * @brief Menyimpan data setiap titik lokasi pada grafik VRP (Depot atau Pelanggan).
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
 * Modifikasi CUDA: Menggunakan array berukuran tetap `tour[MAX_CUSTOMERS]` 
 * agar dapat disimpan di register / Thread-Local Memory GPU dengan cepat.
 */
typedef struct {
    int tour[MAX_CUSTOMERS];
    double total_distance;
    int num_vehicles;
} Solution;

/**
 * @struct SAParameters
 * @brief Konfigurasi hiperparameter untuk algoritma Simulated Annealing.
 */
typedef struct {
    double initial_temp;
    double cooling_rate;
    double min_temp;
    int iterations_per_temp;
} SAParameters;

/* ========================================================================= */
/*              PROTOTIPE FUNGSI UTILITAS & HOST (CPU) (vrp_utils.cu)        */
/* ========================================================================= */

/**
 * @brief Menhitung jarak Euclidean 2D antara dua Node.
 *        Di-inline agar dapat dieksekusi langsung oleh kernel GPU secara efisien.
 */
inline __host__ __device__ double calculate_distance(Node a, Node b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief Membaca file dataset CVRP dari disk (Dijalankan di Host/CPU).
 */
__host__ int read_dataset(const char *filename, Node **nodes_out, int *num_nodes_out, int *capacity_out);

/**
 * @brief Menampilkan rute terperinci per kendaraan ke terminal (Dijalankan di Host/CPU).
 */
__host__ void print_detailed_routes(const Solution *sol, const Node *nodes, int num_customers, int capacity);

/* ========================================================================= */
/*              PROTOTIPE FUNGSI DEVICE (GPU) & SA (vrp_sa.cu)               */
/* ========================================================================= */

/**
 * @brief Mengevaluasi kelayakan rute tour dan menghitung total jarak tempuh pada GPU.
 */
__device__ void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity,
                                 double *out_distance, int *out_vehicles);

/**
 * @brief Menyalin objek Solusi pada thread GPU.
 */
__device__ void copy_solution(Solution *dest, const Solution *src, int num_customers);

/**
 * @brief Membentuk solusi awal acak pada thread GPU menggunakan cuRAND.
 */
__device__ void generate_initial_solution(Solution *sol, int num_customers, curandState *rand_state);

/**
 * @brief Melakukan mutasi rute (Swap / 2-Opt) menggunakan generator cuRAND thread GPU.
 */
__device__ void apply_neighborhood_move(Solution *sol, int num_customers, curandState *rand_state);

/* ========================================================================= */
/*                       KERNEL CUDA & HOST WRAPPER                          */
/* ========================================================================= */

/**
 * @brief Kernel CUDA untuk menginisialisasi state random generator (cuRAND) pada setiap thread.
 * @param states Array pointer cuRAND state di memori GPU.
 * @param seed Nilai seed acak awal.
 * @param total_threads Jumlah total thread GPU.
 */
__global__ void setup_curand_kernel(curandState *states, unsigned long long seed, int total_threads);

/**
 * @brief Kernel utama Simulated Annealing Paralel.
 *        1 Thread CUDA = 1 Rantai Pencarian Simulated Annealing Mandiri.
 * @param d_nodes Array data Node di VRAM (Global Memory).
 * @param d_best_solutions Output solusi terbaik dari masing-masing thread GPU.
 * @param num_customers Jumlah total pelanggan.
 * @param capacity Batas kapasitas kendaraan.
 * @param params Hiperparameter SA.
 * @param d_rand_states State cuRAND per thread.
 */
__global__ void run_simulated_annealing_kernel(const Node *d_nodes, Solution *d_best_solutions,
                                               int num_customers, int capacity,
                                               SAParameters params, curandState *d_rand_states);

/**
 * @brief Fungsi Wrapper Host (CPU) untuk mengelola alokasi VRAM GPU, pemanggilan Kernel, 
 *        dan pengambilan hasil solusi terbaik global (Global Best Reduction).
 * @param h_nodes Array Node di RAM CPU.
 * @param num_customers Jumlah pelanggan.
 * @param capacity Kapasitas kendaraan.
 * @param params Parameter SA.
 * @param num_blocks Jumlah CUDA Block.
 * @param threads_per_block Jumlah Thread per CUDA Block.
 * @param h_global_best_sol Pointer penampung solusi terbaik global di CPU.
 */
__host__ void run_cuda_parallel_sa(const Node *h_nodes, int num_customers, int capacity,
                                   SAParameters params, int num_blocks, int threads_per_block,
                                   Solution *h_global_best_sol);

#endif /* VRP_H */