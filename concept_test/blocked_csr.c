#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitset.h"

// --------------------------- Macro definitions (adjustable based on hardware) ---------------------------
#define BLOCK_SIZE 64  // Block size (recommended 64 to match CPU cache line)
#define N_MAX 100000   // Consistent with csr.c
#define E_MAX 1000000

// --------------------------- Blocked-CSR Structure ---------------------------
typedef struct {
    int *block_row_ptrs;  // Block-level row pointers: block_row_ptrs[b] = starting index of block b's neighbors in block_col_inds
    int *block_offsets;   // Intra-block vertex offsets: block_offsets[b] = first vertex ID of block b
    int *block_col_inds;  // Rearranged neighbor indices within blocks
    int num_blocks;      // Total number of blocks
    int num_vertices;    // Total number of vertices
    int num_edges;       // Total number of edges
} BlockedCSR;

// --------------------------- Helper function: Calculate block ID for a vertex ---------------------------
static int get_block_id(int vertex_id) {
    return vertex_id / BLOCK_SIZE;
}

// --------------------------- 1. Build Blocked-CSR format ---------------------------
BlockedCSR* build_blocked_csr(int num_vertices, const int *csr_row_ptrs, const int *csr_col_inds) {
    BlockedCSR *bcsr = (BlockedCSR*)malloc(sizeof(BlockedCSR));
    if (!bcsr) { perror("malloc BlockedCSR failed"); exit(EXIT_FAILURE); }

    // 1. Initialize basic parameters
    bcsr->num_vertices = num_vertices;
    bcsr->num_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Round up
    bcsr->num_edges = csr_row_ptrs[num_vertices];  // Reuse total edge count from CSR

    // 2. Allocate memory: block_offsets (first vertex ID for each block)
    bcsr->block_offsets = (int*)malloc(bcsr->num_blocks * sizeof(int));
    for (int b = 0; b < bcsr->num_blocks; b++) {
        bcsr->block_offsets[b] = b * BLOCK_SIZE;  // First vertex ID of block b = b*BLOCK_SIZE
    }
    // The last block might be smaller than BLOCK_SIZE, no special handling needed

    // 3. Count total neighbors per block (for building block_row_ptrs)
    int *block_edge_count = (int*)calloc(bcsr->num_blocks, sizeof(int));
    for (int v = 0; v < num_vertices; v++) {
        int block_id = get_block_id(v);
        int edge_count = cs极_row_ptrs[v+1] - csr_row_ptrs[v];
        block_edge_count[block_id] += edge_count;
    }

    // 4. Build block_row_ptrs (prefix sum)
    bcsr->block_row_ptrs = (int*)malloc((bcsr->num_blocks + 1) * sizeof(int));
    bcsr->block_row_ptrs[0] = 0;
    for (int b = 0; b < bcsr->num_blocks; b++) {
        bcsr->block_row_ptrs[b+1] = bcsr->block_row_ptrs[b] + block_edge_count[b];
    }
    free(block_edge_count);

    // 5. Fill block_col_inds (rearrange neighbors by block)
    bcsr->block_col_inds = (int*)malloc(bcsr->num_edges * sizeof(int));
    int *block_write_ptr = (极*)malloc(bcsr->num_blocks * sizeof(int));
    memcpy(block_write_ptr, bcsr->block_row_ptrs, bcsr->num_blocks * sizeof(int));

    for (int v = 0; v < num_vertices; v++) {
        int block_id = get_block_id(v);
        int start = csr_row_ptrs[v];
        int end = csr_row_ptrs[v+1];
        // Copy vertex v's neighbors to its block's block_col_inds region
        for (int i = start; i < end; i++) {
            int neigh = csr_col_inds[i];
            bcsr->block_col_inds[block_write_ptr[block_id]++] = neigh;
        }
    }
    free(block_write_ptr);

    printf("[Blocked-CSR] Build completed: blocks=%d, block size=%d, vertices=%d, edges=%d\n",
           bcsr->num_blocks, BLOCK_SIZE, bcsr->num_vertices, bcsr->num_edges);
    return bcsr;
}

// --------------------------- 2. Blocked-CSR version BFS (adapted for existing push/pull logic) ---------------------------
int bfs_blocked_csr(BlockedCSR *bcsr, int start, int target, int *distances) {
    // Initialize distance array (serial)
    memset(distances, -1, bcsr->num_vertices * sizeof(int));
    distances[start] = 0;

    // Initialize frontier (Bitset, reuse bitset.h)
    bitset_t current_frontier, next_frontier, visited;
    bitset_init(&current_frontier);
    bitset_init(&next_frontier);
    bitset_init(&visited);
    BITSET_SET(&current_frontier, start);
    BITSET_SET(&visited, start);

    // Push-Pull threshold (tunable)
    const double THRESHOLD = 0.05;
    int total_edges = bcsr->num_edges;

    while (1) {
        int frontier_size = bitset_count(&current_frontier);
        if (frontier_size == 0) break;  // No more frontier, target unreachable
        if (distances[target] != -1) break;  // Target found

        // Calculate active edges in current frontier (for density check)
        int active_edges = 0;
        int v = -1;
        while ((v = bitset_next(&current_frontier, v, bcsr->num_vertices)) != -1) {
            // Note: Blocked-CSR lacks vertex-level row pointers, need to find vertex v's block first, then traverse neighbors within block (simplified here, actual implementation requires optimization)
            // [Optimization point]: Pre-store "vertex→intra-block offset" array to avoid traversing all vertices in block
            int block_id = get_block_id(v);
            int block_start = bcsr->block_row_ptrs[block_id];
            int block_end = bcsr->block_row_ptrs[block_id + 1];
            // Traverse all neighbors in block, count neighbors of current vertex v (simplified logic, actual implementation requires storing vertex-level pointers)
            for (int i = block_start; i < block_end; i++) {
                if (bcsr->block_col_inds[i] == v) {  // Assuming neighbor storage includes reverse edges (undirected graph)
                    active_edges++;
                }
            }
        }
        double frontier_density = (double)active_edges / total_edges;

        // Dynamically select Push/Pull mode (core: process in parallel by blocks, improve cache locality)
        if (frontier_density < THRESHOLD) {
            // Push mode: Traverse frontier vertices in parallel by blocks
            #pragma omp parallel num_threads(4)
            {
                int tid = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                // Divide tasks by blocks (each thread processes consecutive blocks, improve cache locality)
                int blocks_per_thread = bcsr->num_blocks / num_threads;
                int start_block = tid * blocks_per_thread;
                int end_block = (tid == num_threads - 1) ? bcsr->num_blocks : start_block + blocks_per_thread;

                for (int b = start_block; b < end_block; b++) {
                    int block_first_v = bcsr->block_offsets[b];
                    int block_last_v = (b == bcsr->num_blocks - 1) ? 
                                     bcsr->num_vertices - 1 : 
                                     block_first_v + BLOCK_SIZE - 1;

                    // Traverse all vertices in block, check if in current frontier
                    int v = block_first_v;
                    while ((v = bitset_next(&current_frontier, v, block_last_v)) != -1) {
                        // Atomically clear current frontier vertex (avoid duplicate processing)
                        #pragma omp critical
                        {
                            BITSET_CLEAR(&current_frontier, v);
                        }

                        // Traverse v's neighbors (neighbors within Blocked-CSR block)
                        int block_start = bcsr->block_row_ptrs[b];
                        int block_end = bcsr->block_row_ptrs[b + 1];
                        for (int i = block_start; i < block_end; i++) {
                            int neigh = bcsr->block_col_inds[i];
                            if (BITSET_TEST(&visited, neigh) == 0) {
                                #pragma omp critical
                                {
                                    if (BITS极_TEST(&visited, neigh) == 0) {
                                        distances[neigh] = distances[v] + 1;
                                        BITSET_SET(&visited, neigh);
                                        BITSET_SET(&next_frontier, neigh);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Pull mode: Traverse all unvisited vertices in parallel by blocks
            #pragma omp parallel for num_threads(4) schedule(dynamic)
            for (int b = 0; b < bcsr->num_blocks; b++) {
                int block_first_v = bcsr->block_offsets[b];
                int block_last_v = (b == bcsr->num_blocks - 1) ? 
                                 bcsr->num_vertices - 1 : 
                                 block_first_v + BLOCK_SIZE - 1;

                // Traverse unvisited vertices in block
                for (极 v = block_first_v; v <= block_last_v; v++) {
                    if (distances[v] != -1) continue;

                    // Check if v's neighbors are in current frontier
                    int min_dist = -1;
                    int block_start = bcsr->block_row_ptrs[b];
                    int block_end = bcsr->block_row_ptrs[b + 1];
                    for (int i = block_start; i < block_end; i++) {
                        int u = bcsr->block_col_inds[i];
                        if (BITSET_TEST(&current_frontier, u)) {
                            min_dist = distances[u] + 1;
                            break;  // Undirected graph, finding one is sufficient
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
                }
            }
        }

        // Iteration: swap frontiers
        BITSET_COPY(&current_frontier, &next_frontier);
        bitset_clear_all(&next_frontier);
    }

    // Free resources
    return distances[target];
}

// --------------------------- 3. Destroy Blocked-CSR ---------------------------
void free_blocked_csr(BlockedCSR *bcsr) {
    if (bcsr) {
        free(bcsr->block_row_ptrs);
        free(bcsr->block_offsets);
        free(bcsr->block_col_inds);
        free(bcsr);
    }
}

// --------------------------- Test entry point (can be integrated into main function) ---------------------------
#ifdef TEST_BLOCKED_CSR
int main() {
    // 1. Get CSR data from csr.c (ensure ROW_PTRS and COL_INDS from csr.c are accessible, or modify to pass as function parameters)
    extern int ROW_PTRS[N_MAX + 1], COL_INDS[E_MAX];
    extern int max_node_id, num_edges;
    if (!read_data_file("facebook_processed.txt")) {  // Assume preprocessed
        return EXIT_FAILURE;
    }
    int num_vertices = max_node_id + 1;
    build_csr(num_vertices);  // Build CSR first

    // 2. Build Blocked-CSR
    BlockedCSR *bcsr = build_blocked_csr(num_vertices, ROW_PTRS, COL_INDS);

    // 3. Execute Blocked-CSR BFS
    int start = 0, target = 100;
    int *distances = (int*)malloc(num_vertices * sizeof(int));
    int distance = bfs_blocked_csr(bcsr, start, target, distances);

    // 4. Output results
    if (distance != -1) {
        printf("Blocked-CSR BFS: Shortest distance from vertex %d to %d = %d\n", start, target, distance);
    } else {
        printf("Blocked-CSR BFS: Vertex %d to %d is unreachable\n", start, target);
    }

    // 5. Free resources
    free(distances);
    free_blocked_csr(bcsr);
    return EXIT_SUCCESS;
}
#endif