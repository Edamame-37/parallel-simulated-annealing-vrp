/**
 * @file vrp_utils.c
 * @brief Implementasi fungsi-fungsi utilitas untuk operasi I/O, kalkulasi jarak Euclidean,
 *        dan pelaporan visual rute kendaraan (CVRP).
 */

#include "vrp.h"

/**
 * @brief Menghitung jarak garis lurus (Euclidean Distance) antara dua titik koordinat.
 *
 * Rumus:
 *   d = sqrt((x_b - x_a)^2 + (y_b - y_a)^2)
 *
 * @param a Titik asal (Node).
 * @param b Titik tujuan (Node).
 * @return Nilai jarak dalam satuan desimal (double).
 */
double calculate_distance(Node a, Node b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt((dx * dx) + (dy * dy));
}

/**
 * @brief Membaca file dataset CVRP dari media penyimpanan dan mem-parsing isinya.
 *
 * Mengabaikan baris kosong dan baris yang diawali dengan '#'.
 * Baris pertama data: [JUMLAH_NODE] [KAPASITAS]
 * Baris berikutnya: [ID] [X] [Y] [DEMAND]
 *
 * @param filename Path file dataset yang akan dibaca.
 * @param nodes_out Output pointer ke array Node yang berhasil dialokasikan di memori.
 * @param num_nodes_out Output jumlah keseluruhan Node yang dibaca (1 Depot + N Pelanggan).
 * @param capacity_out Output kapasitas muatan masing-masing kendaraan.
 * @return 0 jika berhasil membaca dataset, -1 jika berkas tidak ditemukan atau format salah.
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

    /* 1. Membaca header untuk mendapatkan jumlah node dan kapasitas armada */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sscanf(line, "%d %d", &num_nodes, &capacity) == 2) {
            break;
        }
    }

    if (num_nodes <= 1 || capacity <= 0) {
        fprintf(stderr, "[ERROR] Format header tidak valid (nodes: %d, capacity: %d)\n", num_nodes, capacity);
        fclose(fp);
        return -1;
    }

    /* 2. Alokasi array Node untuk menampung seluruh titik */
    Node *nodes = (Node *)malloc(sizeof(Node) * num_nodes);
    if (!nodes) {
        fprintf(stderr, "[ERROR] Gagal mengalokasikan memori untuk array Node.\n");
        fclose(fp);
        return -1;
    }

    /* 3. Membaca data tiap node satu per satu */
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

    /* Verifikasi kelengkapan node */
    if (count != num_nodes) {
        fprintf(stderr, "[ERROR] Jumlah node yang terbaca (%d) berbeda dari header (%d).\n", count, num_nodes);
        free(nodes);
        return -1;
    }

    *nodes_out = nodes;
    *num_nodes_out = num_nodes;
    *capacity_out = capacity;
    return 0;
}

/**
 * @brief Mencetak rincian rute per armada kendaraan secara jelas dan rapi.
 *
 * Format tampilan:
 *   [Kendaraan #K]
 *     Rute: Depot (ID: 0) -> P1(D:x) -> ... -> Depot (ID: 0)
 *     Muatan: M / Q | Sub-Jarak: d
 *
 * @param sol Solusi CVRP yang akan dicetak rinciannya.
 * @param nodes Array data seluruh Node.
 * @param num_customers Jumlah pelanggan.
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

        /* Jika muatan melebihi kapasitas, tutup rute kendaraan saat ini dan kembali ke depot */
        if (current_load + curr.demand > capacity && current_load > 0) {
            route_dist += calculate_distance(prev, depot);
            printf(" -> Depot (ID: 0)\n");
            printf("  Muatan: %d / %d | Sub-Jarak: %.2f\n\n", current_load, capacity, route_dist);

            /* Inisialisasi kendaraan berikutnya */
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

    /* Penutupan rute kendaraan terakhir kembali ke depot */
    route_dist += calculate_distance(prev, depot);
    printf(" -> Depot (ID: 0)\n");
    printf("  Muatan: %d / %d | Sub-Jarak: %.2f\n", current_load, capacity, route_dist);
    printf("==================================================================\n\n");
}
