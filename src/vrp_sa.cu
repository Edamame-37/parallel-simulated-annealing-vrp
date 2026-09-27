/**
 * @file vrp_sa.cu
 * @brief Implementasi logika Simulated Annealing Paralel berbasis CUDA GPU.
 */

#include "../include/vrp.h"

/* ========================================================================= */
/*                      DEVICE HELPER FUNCTIONS (GPU)                        */
/* ========================================================================= */

/**
 * @brief Mengevaluasi kelayakan rute tour dan menghitung total jarak tempuh pada GPU.
 */
__device__ void evaluate_solution(const Solution *sol, const Node *nodes, int num_customers, int capacity,
                                 double *out_distance, int *out_vehicles) {
    double total_dist = 0.0;
    int vehicles = 1;
    int current_load = 0;
    int curr_node_idx = 0; // Mulai dari Depot (indeks 0)

    for (int i = 0; i < num_customers; i++) {
        int cust_idx = sol->tour[i];
        int demand = nodes[cust_idx].demand;

        // Jika muatan melebihi kapasitas, kendaraan kembali ke Depot
        if (current_load + demand > capacity) {
            total_dist += calculate_distance(nodes[curr_node_idx], nodes[0]);
            vehicles++;
            curr_node_idx = 0;
            current_load = 0;
        }

        total_dist += calculate_distance(nodes[curr_node_idx], nodes[cust_idx]);
        current_load += demand;
        curr_node_idx = cust_idx;
    }

    // Kembali ke Depot dari pelanggan terakhir
    total_dist += calculate_distance(nodes[curr_node_idx], nodes[0]);

    *out_distance = total_dist;
    *out_vehicles = vehicles;
}

/**
 * @brief Menyalin isi objek Solution pada thread GPU.
 */
__device__ void copy_solution(Solution *dest, const Solution *src, int num_customers) {
    for (int i = 0; i < num_customers; i++) {
        dest->tour[i] = src->tour[i];
    }
    dest->total_distance = src->total_distance;
    dest->num_vehicles = src->num_vehicles;
}

/**
 * @brief Membentuk solusi awal acak menggunakan pengocokan Fisher-Yates dengan cuRAND.
 */
__device__ void generate_initial_solution(Solution *sol, int num_customers, curandState *rand_state) {
    // Inisialisasi urutan dasar (1 s/d num_customers)
    for (int i = 0; i < num_customers; i++) {
        sol->tour[i] = i + 1;
    }

    // Acak urutan tour (Fisher-Yates Shuffle)
    for (int i = num_customers - 1; i > 0; i--) {
        float r = curand_uniform(rand_state);
        int j = (int)(r * (i + 1));
        if (j > i) j = i; // Guarding boundary overflow

        int temp = sol->tour[i];
        sol->tour[i] = sol->tour[j];
        sol->tour[j] = temp;
    }
}

/**
 * @brief Melakukan mutasi rute (Operator Swap atau 2-Opt Reversal) menggunakan cuRAND.
 */
__device__ void apply_neighborhood_move(Solution *sol, int num_customers, curandState *rand_state) {
    if (num_customers < 2) return;

    // Pilih dua indeks acak berbeda
    int idx1 = (int)(curand_uniform(rand_state) * num_customers);
    int idx2 = (int)(curand_uniform(rand_state) * num_customers);

    if (idx1 >= num_customers) idx1 = num_customers - 1;
    if (idx2 >= num_customers) idx2 = num_customers - 1;

    while (idx1 == idx2) {
        idx2 = (int)(curand_uniform(rand_state) * num_customers);
        if (idx2 >= num_customers) idx2 = num_customers - 1;
    }

    int left = (idx1 < idx2) ? idx1 : idx2;
    int right = (idx1 > idx2) ? idx1 : idx2;

    // Keputusan probabilitas operator mutasi: 50% Swap, 50% 2-Opt (Subarray Reversal)
    float op_choice = curand_uniform(rand_state);

    if (op_choice < 0.5f) {
        // Operator Swap
        int temp = sol->tour[left];
        sol->tour[left] = sol->tour[right];
        sol->tour[right] = temp;
    } else {
        // Operator 2-Opt (Pembalikan segmen rute)
        while (left < right) {
            int temp = sol->tour[left];
            sol->tour[left] = sol->tour[right];
            sol->tour[right] = temp;
            left++;
            right--;
        }
    }
}

/* ========================================================================= */
/*                              KERNEL CUDA                                  */
/* ========================================================================= */

/**
 * @brief Kernel untuk menginisialisasi state cuRAND unik per thread CUDA.
 */
__global__ void setup_curand_kernel(curandState *states, unsigned long long seed, int total_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_threads) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

/**
 * @brief Kernel utama Simulated Annealing Paralel.
 *        1 Thread CUDA = 1 Rantai Pencarian SA Independen.
 */
__global__ void run_simulated_annealing_kernel(const Node *d_nodes, Solution *d_best_solutions,
                                               int num_customers, int capacity,
                                               SAParameters params, curandState *d_rand_states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ambil state cuRAND lokal milik thread
    curandState local_rand_state = d_rand_states[tid];

    // Buat solusi awal dan solusi terbaik lokal thread
    Solution current_sol;
    Solution best_sol;

    generate_initial_solution(&current_sol, num_customers, &local_rand_state);
    evaluate_solution(&current_sol, d_nodes, num_customers, capacity,
                      &current_sol.total_distance, &current_sol.num_vehicles);

    copy_solution(&best_sol, &current_sol, num_customers);

    // Proses Annealing
    double temp = params.initial_temp;

    while (temp > params.min_temp) {
        for (int iter = 0; iter < params.iterations_per_temp; iter++) {
            Solution candidate_sol;
            copy_solution(&candidate_sol, &current_sol, num_customers);

            // Mutasi rute kandidat
            apply_neighborhood_move(&candidate_sol, num_customers, &local_rand_state);
            evaluate_solution(&candidate_sol, d_nodes, num_customers, capacity,
                              &candidate_sol.total_distance, &candidate_sol.num_vehicles);

            double delta = candidate_sol.total_distance - current_sol.total_distance;

            // Kriteria Penerimaan Metropolis
            if (delta < 0.0 || curand_uniform(&local_rand_state) < exp(-delta / temp)) {
                copy_solution(&current_sol, &candidate_sol, num_customers);

                // Perbarui solusi terbaik lokal thread jika menemukan nilai lebih optimal
                if (current_sol.total_distance < best_sol.total_distance) {
                    copy_solution(&best_sol, &current_sol, num_customers);
                }
            }
        }
        temp *= params.cooling_rate;
    }

    // Simpan solusi terbaik thread ke Global Memory VRAM
    d_best_solutions[tid] = best_sol;

    // Simpan kembali state cuRAND terbaru
    d_rand_states[tid] = local_rand_state;
}

/* ========================================================================= */
/*                         HOST WRAPPER FUNCTION                             */
/* ========================================================================= */

/**
 * @brief Wrapper Host (CPU) untuk mengelola memori GPU, memanggil Kernel CUDA,
 *        dan menemukan solusi terbaik global dari seluruh thread GPU.
 */
__host__ void run_cuda_parallel_sa(const Node *h_nodes, int num_customers, int capacity,
                                   SAParameters params, int num_blocks, int threads_per_block,
                                   Solution *h_global_best_sol) {
    int total_threads = num_blocks * threads_per_block;
    int num_nodes = num_customers + 1;

    // Pointer Alokasi Memori VRAM (GPU)
    Node *d_nodes = NULL;
    Solution *d_best_solutions = NULL;
    curandState *d_rand_states = NULL;

    // 1. Alokasi VRAM GPU
    CUDA_CHECK(cudaMalloc((void **)&d_nodes, sizeof(Node) * num_nodes));
    CUDA_CHECK(cudaMalloc((void **)&d_best_solutions, sizeof(Solution) * total_threads));
    CUDA_CHECK(cudaMalloc((void **)&d_rand_states, sizeof(curandState) * total_threads));

    // 2. Transfer Data Node dari RAM Host (CPU) ke VRAM Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes, sizeof(Node) * num_nodes, cudaMemcpyHostToDevice));

    // 3. Inisialisasi State cuRAND pada GPU
    unsigned long long seed = (unsigned long long)time(NULL);
    setup_curand_kernel<<<num_blocks, threads_per_block>>>(d_rand_states, seed, total_threads);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Eksekusi Kernel Simulated Annealing Paralel
    run_simulated_annealing_kernel<<<num_blocks, threads_per_block>>>(
        d_nodes, d_best_solutions, num_customers, capacity, params, d_rand_states
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Transfer Hasil Solusi Seluruh Thread GPU ke CPU
    Solution *h_best_solutions = (Solution *)malloc(sizeof(Solution) * total_threads);
    if (!h_best_solutions) {
        fprintf(stderr, "[ERROR] Gagal alokasi memori array solusi pada CPU.\n");
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_best_solutions, d_best_solutions, sizeof(Solution) * total_threads, cudaMemcpyDeviceToHost));

    // 6. Reduksi Solusi Terbaik Global (Mencari Jarak Minimum pada CPU Host)
    int best_thread_idx = 0;
    double min_distance = h_best_solutions[0].total_distance;

    for (int i = 1; i < total_threads; i++) {
        if (h_best_solutions[i].total_distance < min_distance) {
            min_distance = h_best_solutions[i].total_distance;
            best_thread_idx = i;
        }
    }

    // Salin hasil terbaik ke output host pointer
    *h_global_best_sol = h_best_solutions[best_thread_idx];

    // 7. Pembersihan Memori GPU & Host
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_best_solutions));
    CUDA_CHECK(cudaFree(d_rand_states));
    free(h_best_solutions);
}