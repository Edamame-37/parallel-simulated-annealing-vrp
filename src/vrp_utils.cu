/**
 * @file vrp_utils.cu
 * @brief Implementasi fungsi-fungsi utilitas pendukung I/O dan kalkulasi geometri VRP.
 */

#include "../include/vrp.h"


/* ========================================================================= */
/*                           I/O DATASET (HOST)                              */
/* ========================================================================= */

/**
 * @brief Membaca dataset CVRP dari file teks.
 *        Mendukung format sederhana (baris pertama: jumlah_node kapasitas,
 *        baris berikutnya: id x y demand) serta format standar CVRPLIB.
 */
__host__ int read_dataset(const char *filename, Node **nodes_out, int *num_nodes_out, int *capacity_out) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[ERROR] Tidak dapat membuka file dataset: %s\n", filename);
        return -1;
    }

    char line[256];
    int num_nodes = 0;
    int capacity = 0;
    bool is_cvrplib_format = false;

    // Deteksi awal tipe format file
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "DIMENSION")) {
            sscanf(line, "%*s : %d", &num_nodes);
            if (num_nodes == 0) sscanf(line, "%*s %d", &num_nodes);
            is_cvrplib_format = true;
        } else if (strstr(line, "CAPACITY")) {
            sscanf(line, "%*s : %d", &capacity);
            if (capacity == 0) sscanf(line, "%*s %d", &capacity);
            is_cvrplib_format = true;
        } else if (strstr(line, "NODE_COORD_SECTION")) {
            break;
        }
    }

    // Jika bukan format CVRPLIB, baca dengan format sederhana
    if (!is_cvrplib_format) {
        rewind(fp);
        if (fscanf(fp, "%d %d", &num_nodes, &capacity) != 2) {
            fprintf(stderr, "[ERROR] Format file tidak valid pada baris pertama.\n");
            fclose(fp);
            return -1;
        }
    }

    if (num_nodes <= 0 || capacity <= 0) {
        fprintf(stderr, "[ERROR] Jumlah node (%d) atau kapasitas (%d) tidak valid.\n", num_nodes, capacity);
        fclose(fp);
        return -1;
    }

    // Alokasi memori array Node pada RAM Host
    Node *nodes = (Node *)malloc(sizeof(Node) * num_nodes);
    if (!nodes) {
        fprintf(stderr, "[ERROR] Gagal alokasi memori untuk array Node.\n");
        fclose(fp);
        return -1;
    }

    if (is_cvrplib_format) {
        // 1. Baca Koordinat Node (NODE_COORD_SECTION)
        for (int i = 0; i < num_nodes; i++) {
            if (fscanf(fp, "%d %lf %lf", &nodes[i].id, &nodes[i].x, &nodes[i].y) != 3) {
                fprintf(stderr, "[ERROR] Gagal membaca koordinat node indeks %d.\n", i);
                free(nodes);
                fclose(fp);
                return -1;
            }
        }

        // 2. Cari DEMAND_SECTION
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "DEMAND_SECTION")) break;
        }

        // 3. Baca Demand Pelanggan
        for (int i = 0; i < num_nodes; i++) {
            int dummy_id;
            if (fscanf(fp, "%d %d", &dummy_id, &nodes[i].demand) != 2) {
                fprintf(stderr, "[ERROR] Gagal membaca demand node indeks %d.\n", i);
                free(nodes);
                fclose(fp);
                return -1;
            }
        }
    } else {
        // Format Teks Sederhana: id x y demand
        for (int i = 0; i < num_nodes; i++) {
            if (fscanf(fp, "%d %lf %lf %d", &nodes[i].id, &nodes[i].x, &nodes[i].y, &nodes[i].demand) != 4) {
                fprintf(stderr, "[ERROR] Gagal membaca baris node indeks %d.\n", i);
                free(nodes);
                fclose(fp);
                return -1;
            }
        }
    }

    fclose(fp);

    *nodes_out = nodes;
    *num_nodes_out = num_nodes;
    *capacity_out = capacity;

    return 0;
}

/* ========================================================================= */
/*                         PENCETAKAN HASIL (HOST)                           */
/* ========================================================================= */

/**
 * @brief Menampilkan rincian navigasi setiap kendaraan beserta beban muatan.
 */
__host__ void print_detailed_routes(const Solution *sol, const Node *nodes, int num_customers, int capacity) {
    printf("\n--- RINCIAN RUTE NAVIGASI ARMADA KENDARAAN ---\n");

    int vehicle_count = 1;
    int current_load = 0;
    double route_distance = 0.0;
    int prev_node_idx = 0; // Depot (Node 0)

    printf("\n[Kendaraan #%d]\n", vehicle_count);
    printf("  Rute  : Depot(0)");

    for (int i = 0; i < num_customers; i++) {
        int cust_idx = sol->tour[i];
        int demand = nodes[cust_idx].demand;

        // Cek apakah muatan melebihi kapasitas kendaraan
        if (current_load + demand > capacity) {
            // Kembali ke Depot
            route_distance += calculate_distance(nodes[prev_node_idx], nodes[0]);
            printf(" -> Depot(0)\n");
            printf("  Beban : %d / %d unit\n", current_load, capacity);
            printf("  Jarak : %.4f unit\n", route_distance);

            // Membuka rute kendaraan baru
            vehicle_count++;
            current_load = 0;
            route_distance = 0.0;
            prev_node_idx = 0;

            printf("\n[Kendaraan #%d]\n", vehicle_count);
            printf("  Rute  : Depot(0)");
        }

        printf(" -> Cust %d(ID:%d)", cust_idx, nodes[cust_idx].id);
        route_distance += calculate_distance(nodes[prev_node_idx], nodes[cust_idx]);
        current_load += demand;
        prev_node_idx = cust_idx;
    }

    // Penutupan rute kendaraan terakhir kembali ke Depot
    route_distance += calculate_distance(nodes[prev_node_idx], nodes[0]);
    printf(" -> Depot(0)\n");
    printf("  Beban : %d / %d unit\n", current_load, capacity);
    printf("  Jarak : %.4f unit\n", route_distance);
    printf("------------------------------------------------------------------\n");
}