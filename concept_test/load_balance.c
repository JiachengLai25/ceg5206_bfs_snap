#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bitset.h"

// --------------------------- Structure: Vertex Degree Information ---------------------------
typedef struct {
    int vertex_id;
    int degree;  // Vertex out-degree
} VertexDegree;

// --------------------------- Helper Function: Sort by Degree Descending (for partitioning) ---------------------------
int compare_degree(const void *a, const void *b) {
    VertexDegree *vd1 = (VertexDegree*)a;
    VertexDegree *vd2 = (VertexDegree*)b;
    return vd2->degree - vd1->degree;  // Descending order
}

// --------------------------- 1. Calculate Vertex Degrees (based on CSR) ---------------------------
void compute_vertex_degrees(int num_vertices, const int *csr_row_ptrs, VertexDegree *vd_arr) {
    for (int v = 0; v < num_vertices; v++) {
        vd_arr[v].vertex_id = v;
        vd_arr[v].degree = csr_row_ptrs[v+1] - csr_row_ptrs[v];
    }
    qsort(vd_arr, num_vertices, sizeof(VertexDegree), compare_degree);
    printf("Top 5 vertices by degree: ");
    for (int i = 0; i < 5; i++) {
        printf("(ID=%d, degree=%d) ", vd_arr[i].vertex_id, vd_arr[i].degree);
    }
    printf("\n");
}

// --------------------------- 2. Degree-based Partitioning + Work Stealing BFS (Push Mode) ---------------------------
int bfs_load_balanced(int num_vertices, const int *csr_row_ptrs, const int *csr_col_inds, 
                     int start, int target, int *distances, double *variance_out) {
    // Initialize distance array
    memset(distances, -1, num_vertices * sizeof(int));
    distances[start] = 0;

    // Initialize frontier and visited markers
    bitset_t current_frontier, next_frontier, visited;
    bitset_init(&current_frontier);
    bitset_init(&next_frontier);
    bitset_init(&visited);
    BITSET_SET(&current_frontier, start);
    BITSET_SET(&visited, start);

    // Calculate vertex degrees (for partitioning)
    VertexDegree *vd_arr = (VertexDegree*)malloc(num_vertices * sizeof(VertexDegree));
    compute_vertex_degrees(num_vertices, csr_row_ptrs, vd_arr);

    // Thread edge count statistics (for variance calculation)
    int num_threads = omp_get_max_threads();
    int *edges_processed = (int*)calloc(num_threads, sizeof(int));
    omp_lock_t edges_lock;
    omp_init_lock(&edges_lock);

    // BFS main loop
    const double THRESHOLD = 0.05;
    int total_edges = csr_row_ptrs[num_vertices];

    while (1) {
        int frontier_size = bitset_count(&current_frontier);
        if (frontier_size == 0 || distances[target] != -1) break;

        // Calculate frontier density
        int active_edges = 0;
        int v = -1;
        while ((v = bitset_next(&current_frontier, v, num_vertices)) != -1) {
            active_edges += csr_row_ptrs[v+1] - csr_row_ptrs[v];
        }
        double density = (double)active_edges / total_edges;

        // Reset thread edge count statistics
        memset(edges_processed, 0, num_threads * sizeof(int));

        if (density < THRESHOLD) {
            // Push mode: Degree-based partitioning + work stealing (OpenMP tasks)
            #pragma omp parallel
            {
                #pragma omp single nowait  // Single thread generates tasks, others steal
                {
                    // Process vertices in descending degree order (high-degree vertices first to avoid concentration)
                    for (int i = 0; i < num_vertices; i++) {
                        int v = vd_arr[i].vertex_id;
                        if (BITSET_TEST(&current_frontier, v)) {
                            #pragma omp task firstprivate(v)  // Task-private vertex v
                            {
                                int tid = omp_get_thread_num();
                                int edge_count = 0;  // Edges processed by this thread

                                // Atomically clear frontier vertex
                                #pragma omp critical
                                {
                                    if (BITSET_TEST(&current_frontier, v)) {
                                        BITSET_CLEAR(&current_frontier, v);
                                    } else {
                                        return;  // Already processed by another thread, exit task
                                    }
                                }

                                // Process neighbors
                                int start_idx = csr_row_ptrs[v];
                                int end_idx = csr_row_ptrs[v+1];
                                for (int i = start_idx; i < end_idx; i++) {
                                    int neigh = csr_col_inds[i];
                                    edge_count++;
                                    if (BITSET_TEST(&visited, neigh) == 0) {
                                        #pragma omp critical
                                        {
                                            if (BITSET_TEST(&visited, neigh) == 0) {
                                                distances[neigh] = distances[v] + 1;
                                                BITSET_SET(&visited, neigh);
                                                BITSET_SET(&next_frontier, neigh);
                                            }
                                        }
                                    }
                                }

                                // Count edges processed by this thread
                                omp_set_lock(&edges_lock);
                                edges_processed[tid] += edge_count;
                                omp_unset_lock(&edges_lock);
                            }
                        }
                    }
                }
            }
        } else {
            // Pull mode: Degree-based partitioning (high-degree vertices checked first)
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < num_vertices; i++) {
                int v = vd_arr[i].vertex_id;
                if (distances[v] != -1) continue;

                int tid = omp_get_thread_num();
                int edge_count = 0;
                int min_dist = -1;
                int start_idx = csr_row_ptrs[v];
                int end_idx = csr_row_ptrs[v+1];

                // Check if neighbors are in current frontier
                for (int j = start_idx; j < end_idx; j++) {
                    edge_count++;
                    int u = csr_col_inds[j];
                    if (BITSET_TEST(&current_frontier, u)) {
                        min_dist = distances[u] + 1;
                        break;
                    }
                }

                if (min_dist != -1) {
                    #pragma omp critical
                    {
                        if (distances[v] == -1) {
                            distances[v] = min_dist;
                            BITSET_SET(&visited, v);
                            BITSET_SET(&next_frontier, v);
                        }
                    }
                }

                // Count edges
                omp_set_lock(&edges_lock);
                edges_processed[tid] += edge_count;
                omp_unset_lock(&edges_lock);
            }
        }

        // Calculate load variance (evaluate balance)
        double mean = 0.0;
        for (int t = 0; t < num_threads; t++) {
            mean += edges_processed[t];
        }
        mean /= num_threads;

        double variance = 0.0;
        for (int t = 0; t < num_threads; t++) {
            variance += (edges_processed[t] - mean) * (edges_processed[t] - mean);
        }
        variance /= num_threads;
        *variance_out = variance;  // Output variance
        printf("Current thread count=%d, load variance=%.2f\n", num_threads, variance);

        // Iterate frontier
        BITSET_COPY(&current_frontier, &next_frontier);
        bitset_clear_all(&next_frontier);
    }

    // Free resources
    omp_destroy_lock(&edges_lock);
    free(vd_arr);
    free(edges_processed);
    return distances[target];
}

// --------------------------- Test Entry Point ---------------------------
#ifdef TEST_LOAD_BALANCE
int main() {
    // 1. Read CSR data (need to integrate read_data_file and build_csr from csr.c)
    extern int ROW_PTRS[N_MAX + 1], COL_INDS[E_MAX];
    extern int max_node_id;
    if (!read_data_file("facebook_processed.txt")) {
        return EXIT_FAILURE;
    }
    int num_vertices = max_node_id + 1;
    build_csr(num_vertices);

    // 2. Execute load-balanced BFS
    int start = 0, target = 100;
    int *distances = (int*)malloc(num_vertices * sizeof(int));
    double variance;  // Load variance
    int distance = bfs_load_balanced(num_vertices, ROW_PTRS, COL_INDS, start, target, distances, &variance);

    // 3. Output results
    printf("Load-balanced BFS: Shortest distance from vertex %d to %d = %d, load variance=%.2f\n", 
           start, target, distance, variance);

    free(distances);
    return EXIT_SUCCESS;
}
#endif