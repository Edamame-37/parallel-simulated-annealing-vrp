#ifndef VRP_H
#define VRP_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_CUSTOMERS 256
#define MAX_LINE_LENGTH 1024

// Data structures
typedef struct {
    int id;
    double x;
    double y;
    int demand;
} Node;

typedef struct {
    int tour[MAX_CUSTOMERS]; // Fixed size array to avoid dynamic allocation on GPU
    int num_customers;
    double total_distance;
    double total_demand;
} Solution;

typedef struct {
    double T0;
    double alpha;
    double Tmin;
    int iterations;
    int num_blocks;
    int threads_per_block;
} SAParameters;

// vrp_utils.cu functions
__host__ __device__ double calculate_distance(Node a, Node b);
__host__ void read_dataset(const char* filepath, Node* nodes, int* num_nodes, int* capacity);
__host__ void print_detailed_routes(const Solution* sol, const Node* nodes, int capacity);

// CUDA kernels and host wrappers
__global__ void setup_curand_kernel(curandState* state, unsigned long long seed);
__global__ void run_simulated_annealing_kernel(
    const Node* d_nodes, 
    int num_nodes, 
    int capacity, 
    SAParameters params, 
    curandState* state, 
    Solution* d_best_solutions);
__host__ void run_cuda_parallel_sa(
    const Node* h_nodes, 
    int num_nodes, 
    int capacity, 
    SAParameters params, 
    Solution* h_best_solution);

#endif // VRP_H
