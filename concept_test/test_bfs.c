#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bitset.h"

// Define BFS-related data structures and functions
#define MAX_NODES 5120
#define MAX_EDGES 100000

typedef struct {
    int src;
    int dest;
} Edge;

int ROW_PTRS[MAX_NODES + 1];
int COL_INDS[MAX_EDGES];
Edge edges[MAX_EDGES];
int num_nodes = 0;
int num_edges = 0;
int degrees[MAX_NODES];
int distances[MAX_NODES];

// Declare BFS-related functions
void build_test_graph();
void build_csr();
void bfs(int start_node);
void print_distances();

// Create a test graph: 0-1-2-3-0 forms a cycle, 0-4-5-2 forms another path
void build_test_graph() {
    // Edge list (undirected graph, so each edge is stored twice)
    edges[0].src = 0; edges[0].dest = 1;
    edges[1].src = 1; edges[1].dest = 0;
    edges[2].src = 1; edges[2].dest = 2;
    edges[3].src = 2; edges[3].dest = 1;
    edges[4].src = 2; edges[4].dest = 3;
    edges[5].src = 3; edges[5].dest = 2;
    edges[6].src = 3; edges[6].dest = 0;
    edges[7].src = 0; edges[7].dest = 3;
    edges[8].src = 0; edges[8].dest = 4;
    edges[9].src = 4; edges[9].dest = 0;
    edges[10].src = 4; edges[10].dest = 5;
    edges[11].src = 5; edges[11].dest = 4;
    edges[12].src = 5; edges[12].dest = 2;
    edges[13].src = 2; edges[13].dest = 5;
    
    num_edges = 14;
    num_nodes = 6;
    
    // Initialize degrees
    memset(degrees, 0, sizeof(degrees));
    for (int i = 0; i < num_edges; i++) {
        degrees[edges[i].src]++;
    }
}

// Build CSR representation
void build_csr() {
    // Calculate ROW_PTRS array
    ROW_PTRS[0] = 0;
    for (int i = 1; i < num_nodes + 1; i++) {
        ROW_PTRS[i] = ROW_PTRS[i-1] + degrees[i-1];
    }
    
    // Fill COL_INDS array
    int *temp_ptrs = (int*)malloc(sizeof(int) * num_nodes);
    memcpy(temp_ptrs, ROW_PTRS, sizeof(int) * num_nodes);
    
    for (int i = 0; i < num_edges; i++) {
        int src = edges[i].src;
        int dest = edges[i].dest;
        COL_INDS[temp_ptrs[src]++] = dest;
    }
    
    free(temp_ptrs);
}

// Simplified BFS implementation
void bfs(int start_node) {
    // Initialize distance array
    for (int i = 0; i < num_nodes; i++) {
        distances[i] = -1;
    }
    
    // Use queue for BFS
    int queue[MAX_NODES];
    int front = 0, rear = 0;
    
    // Enqueue start node
    queue[rear++] = start_node;
    distances[start_node] = 0;
    
    while (front < rear) {
        int current = queue[front++];
        
        // Iterate through all neighbors
        for (int i = ROW_PTRS[current]; i < ROW_PTRS[current + 1]; i++) {
            int neighbor = COL_INDS[i];
            
            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[current] + 1;
                queue[rear++] = neighbor;
            }
        }
    }
}

// Print distance information
void print_distances() {
    printf("Node distance information:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: %d\n", i, distances[i]);
    }
}

// Test BFS from different starting nodes
void test_bfs_from_different_starts() {
    build_test_graph();
    build_csr();
    
    printf("Testing BFS from different start nodes...\n");
    
    // Test BFS starting from node 0
    printf("\nBFS starting from node 0:\n");
    bfs(0);
    print_distances();
    
    // Verify if distances are correct
    int expected_distances0[] = {0, 1, 2, 1, 1, 2};
    int pass = 1;
    for (int i = 0; i < num_nodes; i++) {
        if (distances[i] != expected_distances0[i]) {
            pass = 0;
            printf("✗ Node %d distance error: expected %d, actual %d\n", i, expected_distances0[i], distances[i]);
        }
    }
    
    if (pass) {
        printf("✓ BFS distance calculation starting from node 0 is correct\n");
    }
    
    // Test BFS starting from node 5
    printf("\nBFS starting from node 5:\n");
    bfs(5);
    print_distances();
    
    // Verify if distances are correct
    int expected_distances5[] = {2, 2, 1, 2, 1, 0};
    pass = 1;
    for (int i = 0; i < num_nodes; i++) {
        if (distances[i] != expected_distances5[i]) {
            pass = 0;
            printf("✗ Node %d distance error: expected %d, actual %d\n", i, expected_distances5[i], distances[i]);
        }
    }
    
    if (pass) {
        printf("✓ BFS distance calculation starting from node 5 is correct\n");
    }
}

// Test isolated nodes
void test_isolated_node() {
    printf("\nTesting isolated node...\n");
    
    // Reset graph, add an isolated node 6
    num_edges = 0;
    num_nodes = 7;
    
    // Initialize degrees
    memset(degrees, 0, sizeof(degrees));
    
    // Only add edge between node 0 and 1
    edges[num_edges].src = 0;
    edges[num_edges].dest = 1;
    degrees[0]++;
    num_edges++;
    
    edges[num_edges].src = 1;
    edges[num_edges].dest = 0;
    degrees[1]++;
    num_edges++;
    
    build_csr();
    
    // BFS starting from node 0
    bfs(0);
    print_distances();
    
    // Verify isolated node 6 has distance -1
    if (distances[6] == -1) {
        printf("✓ Isolated node detection is correct\n");
    } else {
        printf("✗ Isolated node detection is incorrect\n");
    }
}

// Test Push-Pull mode switching logic
void test_push_pull_switch() {
    printf("\nTesting Push-Pull mode switching logic...\n");
    
    // Create a small graph for testing
    build_test_graph();
    build_csr();
    
    // Simulate Push-Pull switching decision
    // When the number of active nodes is less than a certain proportion of total nodes, should switch to Pull mode
    int total_nodes = num_nodes;
    int active_nodes_small = 1;  // Few active nodes, should use Pull mode
    int active_nodes_large = total_nodes / 2;  // Many active nodes, should use Push mode
    
    float threshold = 0.1f;  // Assume threshold is 10%
    
    printf("Active nodes: %d, Total nodes: %d, Ratio: %.2f\n", 
           active_nodes_small, total_nodes, (float)active_nodes_small / total_nodes);
    if ((float)active_nodes_small / total_nodes < threshold) {
        printf("✓ Correct decision: using Pull mode\n");
    } else {
        printf("✗ Incorrect decision\n");
    }
    
    printf("Active nodes: %d, Total nodes: %d, Ratio: %.2f\n", 
           active_nodes_large, total_nodes, (float)active_nodes_large / total_nodes);
    if ((float)active_nodes_large / total_nodes >= threshold) {
        printf("✓ Correct decision: using Push mode\n");
    } else {
        printf("✗ Incorrect decision\n");
    }
}

int main() {
    printf("Starting BFS unit tests...\n\n");
    
    test_bfs_from_different_starts();
    test_isolated_node();
    test_push_pull_switch();
    
    printf("\nBFS unit tests completed!\n");
    return 0;
}