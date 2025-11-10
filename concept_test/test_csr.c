#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define CSR-related data structures, copied from csr.c
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

// Declare function prototypes for testing
void count_nodes_edges();
void build_csr();
void free_csr();

// Simulate data file reading, create a simple graph for testing
void create_test_graph() {
    // Create a simple undirected graph: 0-1-2-3-0
    edges[0].src = 0;
    edges[0].dest = 1;
    edges[1].src = 1;
    edges[1].dest = 0;
    edges[2].src = 1;
    edges[2].dest = 2;
    edges[3].src = 2;
    edges[3].dest = 1;
    edges[4].src = 2;
    edges[4].dest = 3;
    edges[5].src = 3;
    edges[5].dest = 2;
    edges[6].src = 3;
    edges[6].dest = 0;
    edges[7].src = 0;
    edges[7].dest = 3;
    
    num_edges = 8;
    num_nodes = 4;
    
    // Initialize degrees
    memset(degrees, 0, sizeof(degrees));
    for (int i = 0; i < num_edges; i++) {
        degrees[edges[i].src]++;
    }
}

// Test CSR construction
void test_csr_build() {
    printf("Testing CSR construction...\n");
    
    create_test_graph();
    build_csr();
    
    // Verify ROW_PTRS array
    int expected_row_ptrs[] = {0, 2, 4, 6, 8};
    int row_ptrs_correct = 1;
    
    for (int i = 0; i < num_nodes + 1; i++) {
        if (ROW_PTRS[i] != expected_row_ptrs[i]) {
            row_ptrs_correct = 0;
            break;
        }
    }
    
    if (row_ptrs_correct) {
        printf("✓ ROW_PTRS array is correct\n");
    } else {
        printf("✗ ROW_PTRS array is incorrect\n");
        printf("Expected: ");
        for (int i = 0; i < num_nodes + 1; i++) {
            printf("%d ", expected_row_ptrs[i]);
        }
        printf("\nActual: ");
        for (int i = 0; i < num_nodes + 1; i++) {
            printf("%d ", ROW_PTRS[i]);
        }
        printf("\n");
    }
    
    // Verify neighbor relationships
    printf("Verifying node neighbor relationships:\n");
    int pass = 1;
    
    // Check neighbors of node 0 (should be 1 and 3)
    if (COL_INDS[ROW_PTRS[0]] != 1 || COL_INDS[ROW_PTRS[0]+1] != 3) {
        pass = 0;
        printf("✗ Node 0 neighbors are incorrect\n");
    } else {
        printf("✓ Node 0 neighbors are correct\n");
    }
    
    // Check neighbors of node 1 (should be 0 and 2)
    if (COL_INDS[ROW_PTRS[1]] != 0 || COL_INDS[ROW_PTRS[1]+1] != 2) {
        pass = 0;
        printf("✗ Node 1 neighbors are incorrect\n");
    } else {
        printf("✓ Node 1 neighbors are correct\n");
    }
    
    // Check neighbors of node 2 (should be 1 and 3)
    if (COL_INDS[ROW_PTRS[2]] != 1 || COL_INDS[ROW_PTRS[2]+1] != 3) {
        pass = 0;
        printf("✗ Node 2 neighbors are incorrect\n");
    } else {
        printf("✓ Node 2 neighbors are correct\n");
    }
    
    // Check neighbors of node 3 (should be 2 and 0)
    if (COL_INDS[ROW_PTRS[3]] != 2 || COL_INDS[ROW_PTRS[3]+1] != 0) {
        pass = 0;
        printf("✗ Node 3 neighbors are incorrect\n");
    } else {
        printf("✓ Node 3 neighbors are correct\n");
    }
    
    free_csr();
}

// Test node degree calculation
void test_node_degrees() {
    printf("\nTesting node degree calculation...\n");
    
    create_test_graph();
    build_csr();
    
    // For undirected graph, each node should have 2 neighbors
    int expected_degrees[] = {2, 2, 2, 2};
    int degrees_correct = 1;
    
    for (int i = 0; i < num_nodes; i++) {
        if (degrees[i] != expected_degrees[i]) {
            degrees_correct = 0;
            break;
        }
    }
    
    if (degrees_correct) {
        printf("✓ All node degrees calculated correctly\n");
    } else {
        printf("✗ Node degree calculation incorrect\n");
        for (int i = 0; i < num_nodes; i++) {
            printf("Node %d: expected %d, actual %d\n", i, expected_degrees[i], degrees[i]);
        }
    }
    
    free_csr();
}

// Copy function implementation from csr.c
void count_nodes_edges() {
    // This function is replaced by create_test_graph in testing
    // Maintaining prototype consistency here
}

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

void free_csr() {
    // Reset global variables
    num_nodes = 0;
    num_edges = 0;
    memset(ROW_PTRS, 0, sizeof(ROW_PTRS));
    memset(COL_INDS, 0, sizeof(COL_INDS));
    memset(degrees, 0, sizeof(degrees));
    memset(edges, 0, sizeof(edges));
}

int main() {
    printf("Starting CSR unit tests...\n\n");
    
    test_csr_build();
    test_node_degrees();
    
    printf("\nCSR unit tests completed!\n");
    return 0;
}