#include "../include/vrp.h"
#include <string.h>

__host__ __device__ double calculate_distance(Node a, Node b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

__host__ void read_dataset(const char* filepath, Node* nodes, int* num_nodes, int* capacity) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Error opening dataset file: %s\n", filepath);
        exit(EXIT_FAILURE);
    }
    
    char line[MAX_LINE_LENGTH];
    *num_nodes = 0;
    *capacity = 0;
    int in_node_coord_section = 0;
    int in_demand_section = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "CAPACITY", 8) == 0) {
            sscanf(line, "CAPACITY : %d", capacity);
        } else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0) {
            in_node_coord_section = 1;
            in_demand_section = 0;
        } else if (strncmp(line, "DEMAND_SECTION", 14) == 0) {
            in_demand_section = 1;
            in_node_coord_section = 0;
        } else if (strncmp(line, "DEPOT_SECTION", 13) == 0 || strncmp(line, "EOF", 3) == 0) {
            break; 
        } else {
            if (in_node_coord_section) {
                int id;
                double x, y;
                if (sscanf(line, "%d %lf %lf", &id, &x, &y) == 3) {
                    nodes[*num_nodes].id = id;
                    nodes[*num_nodes].x = x;
                    nodes[*num_nodes].y = y;
                    nodes[*num_nodes].demand = 0;
                    (*num_nodes)++;
                }
            } else if (in_demand_section) {
                int id, demand;
                if (sscanf(line, "%d %d", &id, &demand) == 2) {
                    for(int i = 0; i < *num_nodes; i++) {
                        if (nodes[i].id == id) {
                            nodes[i].demand = demand;
                            break;
                        }
                    }
                }
            }
        }
    }
    fclose(file);
}

__host__ void print_detailed_routes(const Solution* sol, const Node* nodes, int capacity) {
    printf("Best Route:\n");
    for (int i = 0; i < sol->num_customers; i++) {
        printf("%d ", sol->tour[i]);
    }
    printf("\nTotal Distance: %.2f\n", sol->total_distance);
}
