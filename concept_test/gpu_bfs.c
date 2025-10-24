#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bitset.h"  // Reuse host-side Bitset logic

// --------------------------- Macro Definitions (Adapted for GPU Architecture) ---------------------------
#define BLOCK_SIZE 256    // Thread block size
#define WARP_SIZE 32      // GPU Warp size (fixed)
#define SHMEM_NEIGH_SIZE 1024  // Shared memory neighbor cache size

// --------------------------- CUDA Error Checking Macro ---------------------------
#define CHECK_CUDA_ERR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --------------------------- 1. GPU Kernel Function: Push-mode BFS (Warp Cooperative) ---------------------------
__global__ void bfs_gpu_push(const int *d_row_ptrs, const int *d_col_inds, 
                           int *d_dist, const int *d_curr_frontier, int curr_size,
                           int *d_new_frontier, int *d_new_size) {
    // Shared memory: Cache neighbors of vertices processed by current block (reduce global memory access)
    __shared__ int shmem_neighs[SHMEM_NEIGH_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= curr_size) return;

    // 1. Each thread processes one frontier vertex
    int u = d_curr_frontier[tid];
    int start = d_row_ptrs[u];
    int end = d_row_ptrs[u + 1];
    int neigh_count = end - start;

    // 2. Warp cooperation: Cache vertex u's neighbors to shared memory (memory coalescing)
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int shmem_offset = warp_id * WARP_SIZE;  // Each warp gets exclusive shared memory region

    if (lane_id < neigh_count) {
        shmem_neighs[shmem_offset + lane_id] = d_col_inds[start + lane_id];
    }
    __syncthreads();  // Wait for all threads in warp to complete caching

    // 3. Traverse neighbors, atomically update distances and new frontier
    for (int i = 0; i < neigh_count; i++) {
        int v = shmem_neighs[shmem_offset + i];
        // Atomic CAS: Ensure only one thread marks v's distance (avoid race condition)
        if (atomicCAS(&d_dist[v], -1, d_dist[u] + 1) == -1) {
            // Atomically add to new frontier
            int pos = atomicAdd(d_new_size, 1);
            d_new_frontier[pos] = v;
        }
    }
}

// --------------------------- 2. Host-side GPU BFS Entry Point ---------------------------
int gpu_bfs(int start, int target, int num_vertices, const int *h_row_ptrs, const int *h_col_inds, int num_edges) {
    // 1. Device memory allocation
    int *d_row_ptrs, *d_col_inds, *d_dist;
    int *d_curr_frontier, *d_new_frontier, *d_new_size;

    CHECK_CUDA_ERR(cudaMalloc(&d_row_ptrs, (num_vertices + 1) * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc(&d_col_inds, num_edges * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc(&极_dist, num_vertices * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc(&d_curr_frontier, num_vertices * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc(&d_new_frontier, num_vertices * sizeof(int)));
    CHECK_CUDA_ERR(cudaMalloc(&d_new_size, sizeof(int)));

    // 2. Data copy (host→device)
    CHECK_CUDA_ERR(cudaMemcpy(d_row_ptrs, h_row_ptrs, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_col_inds, h_col_inds, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemset(d_dist, -1, num_vertices * sizeof(int)));  // Initialize distances to -1
    CHECK_CUDA_ERR(cudaMemcpy(&d_dist[start], &(int){0}, sizeof(int), cudaMemcpyHostToDevice));  // Start vertex distance 0
    CHECK_CUDA_ERR(cudaMemcpy(d_curr_frontier, &start, sizeof(int), cudaMemcpyHostToDevice));  // Initial frontier

    // 3. GPU configuration
    int curr_size = 1;
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((curr_size + block_dim.x - 1) / block_dim.x);

    // 4. BFS main loop
    while (curr_size > 0) {
        // Reset new frontier size
        CHECK_CUDA_ERR(cudaMemset(d_new_size, 0, sizeof(int)));

        // Launch kernel
        bfs_gpu_push<<<grid_dim, block_dim>>>(d_row_ptrs, d_col_inds, d_dist,
                                           d_curr_frontier, curr_size, d_new_frontier, d_new_size);
        CHECK_CUDA_ERR(cudaGetLastError());  // Check kernel launch error
        CHECK_CUDA_ERR(cudaDeviceSynchronize());  // Wait for kernel completion

        // Copy new frontier size to host
        CHECK_CUDA_ERR(cudaMemcpy(&curr_size, d_new_size, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap frontier queues (avoid repeated allocation)
        int *tmp = d_curr_frontier;
        d_curr_frontier = d_new_frontier;
        d_new_frontier = tmp;

        // Check if target found (early termination)
        int target_dist;
        CHECK_CUDA_ERR(cudaMemcpy(&target_dist, &d_dist[target], sizeof(int), cudaMemcpyDeviceToHost));
        if (target_dist != -1) break;
    }

    // 5. Copy result and free resources
    int result;
    CHECK_CUDA_ERR(cudaMemcpy(&result, &d_dist[target], sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_row_ptrs);
    cudaFree(d_col_inds);
    cudaFree(d_dist);
    cudaFree(d_curr_frontier);
    cudaFree(d_new_frontier);
    cudaFree(d_new_size);

    return result;
}

// --------------------------- Test Entry Point ---------------------------
int main(int argc, char **argv) {
    // 1. Read CSR data (need to integrate csr.c logic, or pass h_row_ptrs and h_col_inds via parameters)
    extern int ROW_PTRS[N_MAX + 1], COL_INDS[E_MAX];
    extern int max_node_id, num_edges;
    if (!read_data_file("facebook_processed.txt")) {
        return EXIT_FAILURE;
    }
    int num_vertices = max_node_id + 1;
    build_csr(num_vertices);

    // 2. Execute GPU BFS
    int start = 0, target = 100;
    printf("GPU BFS: Calculating shortest distance from vertex %d to %d...\n", start, target);

    // Timing
    cudaEvent_t start_event, end_event;
    CHECK_CUDA_ERR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERR(cudaEventCreate(&end_event));
    CHECK_CUDA_ERR(cudaEventRecord(start_event));

    int distance = gpu_bfs(start, target, num_vertices, ROW_PTRS, COL_INDS, num_edges);

    CHECK_CUDA_ERR(cudaEventRecord(end_event));
    CHECK_CUDA_ERR(cudaEventSynchronize(end_event));
    float elapsed_ms;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&elapsed_ms极, start_event, end_event));

    // 3. Output results
    if (distance != -1) {
        printf("GPU BFS Result: Shortest distance=%d, Time=%.2f ms, TEPS=%.2f MTEPS\n",
               distance, elapsed_ms, (num_edges / (elapsed_ms / 1000)) / 1e6);
    } else {
        printf("GPU BFS Result: Vertex %d to %d is unreachable\n", start, target);
    }

    // Free CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    return EXIT_SUCCESS;
}