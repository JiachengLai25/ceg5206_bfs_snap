#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// ========================
// 1. 图的数据结构定义 (Adjacency List)
// ========================

typedef struct Node {
    int dest;
    struct Node* next;
} Node;

typedef struct Graph {
    int numVertices;
    Node** adjLists;
} Graph;

// ========================
// 2. 队列的数据结构定义 (用于 BFS)
// ========================

typedef struct QNode {
    int data;
    struct QNode* next;
} QNode;

typedef struct Queue {
    QNode *front;
    QNode *rear;
} Queue;

// ========================
// 3. 辅助函数 (创建节点, 队列操作等)
// ========================

Node* createNode(int dest) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) { perror("malloc failed"); exit(EXIT_FAILURE); }
    newNode->dest = dest;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int vertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    if (graph == NULL) { perror("malloc failed"); exit(EXIT_FAILURE); }
    graph->numVertices = vertices;

    graph->adjLists = (Node**)calloc(vertices, sizeof(Node*)); // 使用 calloc 初始化为 NULL
    if (graph->adjLists == NULL) { perror("calloc failed"); exit(EXIT_FAILURE); }

    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    if (src >= graph->numVertices || dest >= graph->numVertices) {
        // 错误处理: ID超出范围，通常表示预扫描阶段出错或文件包含无效 ID
        fprintf(stderr, "Warning: Edge (%d, %d) skipped, ID out of bounds.\n", src, dest);
        return;
    }
    
    // 从 src 到 dest
    Node* newNode = createNode(dest);
    newNode->next = graph->adjLists[src];
    graph->adjLists[src] = newNode;

    // 从 dest 到 src (无向图)
    newNode = createNode(src);
    newNode->next = graph->adjLists[dest];
    graph->adjLists[dest] = newNode;
}

// (Queue functions - enqueue, dequeue, isEmpty - 保持不变，此处省略以节省空间)
// ... (你可以把上一个回答中的队列函数复制过来) ...
Queue* createQueue() {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    if (q == NULL) { perror("malloc failed"); exit(EXIT_FAILURE); }
    q->front = q->rear = NULL;
    return q;
}

void enqueue(Queue* q, int value) {
    QNode* newNode = (QNode*)malloc(sizeof(QNode));
    if (newNode == NULL) { perror("malloc failed"); exit(EXIT_FAILURE); }
    newNode->data = value;
    newNode->next = NULL;
    
    if (q->rear == NULL) {
        q->front = q->rear = newNode;
        return;
    }
    
    q->rear->next = newNode;
    q->rear = newNode;
}

int dequeue(Queue* q) {
    if (q->front == NULL)
        return -1;

    QNode* temp = q->front;
    int value = temp->data;
    q->front = q->front->next;
    
    if (q->front == NULL)
        q->rear = NULL;

    free(temp);
    return value;
}

int isEmpty(Queue* q) {
    return q->front == NULL;
}

/**
 * @brief 执行广度优先搜索 (BFS) 并计算最短距离。
 * @return int 返回目标节点的距离，如果不可达返回 -1。
 */
int bfs_shortest_distance(Graph* graph, int startNode, int targetNode, int* distances) {
    // 检查起始和目标节点是否在图中存在
    if (startNode >= graph->numVertices || targetNode >= graph->numVertices || startNode < 0 || targetNode < 0) {
        fprintf(stderr, "Error: Start/Target user ID is outside the calculated graph range [0, %d].\n", graph->numVertices - 1);
        return -1;
    }
    
    for (int i = 0; i < graph->numVertices; i++) {
        distances[i] = -1;
    }

    Queue* q = createQueue();
    enqueue(q, startNode);
    distances[startNode] = 0;

    while (!isEmpty(q)) {
        int currentNode = dequeue(q);

        if (currentNode == targetNode) {
            // 找到目标，释放队列并返回距离
            // 注意: 完整的内存清理需要释放所有 QNode*，但这里从简
            return distances[targetNode]; 
        }

        Node* temp = graph->adjLists[currentNode];
        while (temp) {
            int neighbor = temp->dest;

            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[currentNode] + 1;
                enqueue(q, neighbor);
            }
            temp = temp->next;
        }
    }

    // 目标不可达
    return -1;
}

// 核心 I/O 函数：现在只需要一次读取即可
Graph* read_c_friendly_data(const char* filename, int* num_vertices) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening preprocessed file");
        return NULL;
    }

    // 1. 读取第一行的节点总数
    if (fscanf(file, "%d", num_vertices) != 1) {
        fprintf(stderr, "Error: Could not read total number of vertices from the first line.\n");
        fclose(file);
        return NULL;
    }
    
    printf("Read total vertices: %d\n", *num_vertices);

    // 2. 创建图结构
    Graph* graph = createGraph(*num_vertices);
    
    // 3. 读取边列表
    printf("Building adjacency list from edges...\n");
    int u, v;
    int edge_count = 0;
    
    // 循环读取文件中的 ID 对
    while (fscanf(file, "%d %d", &u, &v) == 2) {
        // 由于 Python 已经处理了映射，这里的 u 和 v 都是从 0 开始的连续 ID
        addEdge(graph, u, v);
        edge_count++;
    }

    fclose(file);
    printf("Graph built successfully. Total Edges added: %d\n", edge_count);
    
    return graph;
}

// 释放图结构占用的内存
void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->numVertices; i++) {
        Node* current = graph->adjLists[i];
        while (current != NULL) {
            Node* next = current->next;
            free(current);
            current = next;
        }
    }
    free(graph->adjLists);
    free(graph);
}

// ========================
// 5. 主程序
// ========================

int main(int argc, char *argv[]) {
    // 现在 C 程序接收预处理后的文件，以及映射后的 start/target ID
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <preprocessed_file> <mapped_start_id> <mapped_target_id>\n", argv[0]);
        fprintf(stderr, "Use 'python preprocess.py' first to generate the file and get the mapped IDs.\n");
        return 1;
    }

    const char* filename = argv[1];
    // 接收映射后的 ID
    int startUser = atoi(argv[2]); 
    int targetUser = atoi(argv[3]);
    int num_vertices = 0;

    // 1. 读取文件并构建图
    Graph* socialGraph = read_c_friendly_data(filename, &num_vertices);

    if (socialGraph == NULL) {
        return 1;
    }

    // 2. 检查用户ID是否在有效范围内
    if (startUser >= num_vertices || targetUser >= num_vertices || startUser < 0 || targetUser < 0) {
        fprintf(stderr, "Error: Start/Target ID out of the mapped range [0, %d]. Check your mapped IDs.\n", num_vertices - 1);
        freeGraph(socialGraph);
        return 1;
    }

    // 3. 分配距离数组
    int* distances = (int*)malloc(socialGraph->numVertices * sizeof(int));
    if (distances == NULL) { perror("malloc failed"); freeGraph(socialGraph); return 1; }

    // 4. 执行 BFS 查找社交距离
    printf("\n--- Shortest Distance Calculation ---\n");
    printf("Finding distance from Mapped ID %d to Mapped ID %d...\n", startUser, targetUser);
    
    int distance = bfs_shortest_distance(socialGraph, startUser, targetUser, distances);

    // 5. 输出结果
    if (distance != -1) {
        printf("\nRESULT: Social Distance (Shortest Path Length) is: %d\n", distance);
    } else {
        printf("\nRESULT: Mapped ID %d and Mapped ID %d are unreachable.\n", startUser, targetUser);
    }
    
    // 6. 清理内存
    free(distances);
    freeGraph(socialGraph);

    return 0;
}