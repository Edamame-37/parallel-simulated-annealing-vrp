#include "../include/vrp.h"

int main(int argc, char** argv) {
    if (argc != 8) {
        printf("Usage: %s <dataset> <T0> <Alpha> <Tmin> <Iterations> <Num_Blocks> <Threads_Per_Block>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    const char* dataset = argv[1];
    SAParameters params;
    params.T0 = atof(argv[2]);
    params.alpha = atof(argv[3]);
    params.Tmin = atof(argv[4]);
    params.iterations = atoi(argv[5]);
    params.num_blocks = atoi(argv[6]);
    params.threads_per_block = atoi(argv[7]);
    
    Node h_nodes[MAX_CUSTOMERS];
    int num_nodes;
    int capacity;
    
    read_dataset(dataset, h_nodes, &num_nodes, &capacity);
    
    printf("Dataset: %s\n", dataset);
    printf("Nodes: %d, Capacity: %d\n", num_nodes, capacity);
    
    Solution global_best;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    run_cuda_parallel_sa(h_nodes, num_nodes, capacity, params, &global_best);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\nExecution Time: %.3f ms\n", milliseconds);
    print_detailed_routes(&global_best, h_nodes, capacity);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return EXIT_SUCCESS;
}
