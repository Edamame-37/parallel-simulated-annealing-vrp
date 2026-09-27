#include "../include/vrp.h"

__device__ void evaluate_solution(Solution* sol, const Node* nodes, int capacity) {
    sol->total_distance = 0.0;
    double current_load = 0;
    int last_node_idx = 0; 
    
    for (int i = 0; i < sol->num_customers; i++) {
        int next_node_idx = sol->tour[i];
        
        if (current_load + nodes[next_node_idx].demand > capacity) {
            sol->total_distance += calculate_distance(nodes[last_node_idx], nodes[0]);
            current_load = 0;
            last_node_idx = 0;
        }
        
        sol->total_distance += calculate_distance(nodes[last_node_idx], nodes[next_node_idx]);
        current_load += nodes[next_node_idx].demand;
        last_node_idx = next_node_idx;
    }
    
    sol->total_distance += calculate_distance(nodes[last_node_idx], nodes[0]);
}

__device__ void copy_solution(Solution* dest, const Solution* src) {
    dest->num_customers = src->num_customers;
    dest->total_distance = src->total_distance;
    dest->total_demand = src->total_demand;
    for (int i = 0; i < src->num_customers; i++) {
        dest->tour[i] = src->tour[i];
    }
}

__device__ void generate_initial_solution(Solution* sol, int num_nodes, curandState* state) {
    sol->num_customers = num_nodes - 1;
    for (int i = 0; i < sol->num_customers; i++) {
        sol->tour[i] = i + 1;
    }
    
    // Shuffle
    for (int i = sol->num_customers - 1; i > 0; i--) {
        int j = curand(state) % (i + 1);
        int temp = sol->tour[i];
        sol->tour[i] = sol->tour[j];
        sol->tour[j] = temp;
    }
}

__device__ void apply_neighborhood_move(Solution* current, Solution* next, curandState* state) {
    copy_solution(next, current);
    if (next->num_customers < 2) return;
    
    int i = curand(state) % next->num_customers;
    int j = curand(state) % next->num_customers;
    
    int temp = next->tour[i];
    next->tour[i] = next->tour[j];
    next->tour[j] = temp;
}

__global__ void setup_curand_kernel(curandState* state, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void run_simulated_annealing_kernel(
    const Node* d_nodes, 
    int num_nodes, 
    int capacity, 
    SAParameters params, 
    curandState* state, 
    Solution* d_best_solutions) 
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState local_state = state[id];
    
    Solution current_sol;
    Solution next_sol;
    Solution best_sol;
    
    generate_initial_solution(&current_sol, num_nodes, &local_state);
    evaluate_solution(&current_sol, d_nodes, capacity);
    copy_solution(&best_sol, &current_sol);
    
    double T = params.T0;
    
    while (T > params.Tmin) {
        for (int i = 0; i < params.iterations; i++) {
            apply_neighborhood_move(&current_sol, &next_sol, &local_state);
            evaluate_solution(&next_sol, d_nodes, capacity);
            
            double delta = next_sol.total_distance - current_sol.total_distance;
            
            if (delta < 0) {
                copy_solution(&current_sol, &next_sol);
                if (current_sol.total_distance < best_sol.total_distance) {
                    copy_solution(&best_sol, &current_sol);
                }
            } else {
                double r = curand_uniform(&local_state);
                if (r < exp(-delta / T)) {
                    copy_solution(&current_sol, &next_sol);
                }
            }
        }
        T *= params.alpha;
    }
    
    state[id] = local_state;
    copy_solution(&d_best_solutions[id], &best_sol);
}

__host__ void run_cuda_parallel_sa(
    const Node* h_nodes, 
    int num_nodes, 
    int capacity, 
    SAParameters params, 
    Solution* h_best_solution) 
{
    int total_threads = params.num_blocks * params.threads_per_block;
    
    Node* d_nodes;
    cudaMalloc((void**)&d_nodes, num_nodes * sizeof(Node));
    cudaMemcpy(d_nodes, h_nodes, num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    
    curandState* d_state;
    cudaMalloc((void**)&d_state, total_threads * sizeof(curandState));
    
    Solution* d_best_solutions;
    cudaMalloc((void**)&d_best_solutions, total_threads * sizeof(Solution));
    
    setup_curand_kernel<<<params.num_blocks, params.threads_per_block>>>(d_state, 1234ULL);
    cudaDeviceSynchronize();
    
    run_simulated_annealing_kernel<<<params.num_blocks, params.threads_per_block>>>(
        d_nodes, num_nodes, capacity, params, d_state, d_best_solutions
    );
    cudaDeviceSynchronize();
    
    Solution* h_all_solutions = (Solution*)malloc(total_threads * sizeof(Solution));
    cudaMemcpy(h_all_solutions, d_best_solutions, total_threads * sizeof(Solution), cudaMemcpyDeviceToHost);
    
    h_best_solution->total_distance = 1e9; 
    for (int i = 0; i < total_threads; i++) {
        if (h_all_solutions[i].total_distance < h_best_solution->total_distance) {
            *h_best_solution = h_all_solutions[i];
        }
    }
    
    free(h_all_solutions);
    cudaFree(d_nodes);
    cudaFree(d_state);
    cudaFree(d_best_solutions);
}
