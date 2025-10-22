#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "bitset.h"

void push_mode(int numVertices, bitset_t* current_frontier, bitset_t* next_frontier, 
               int* distances, bitset_t* visited, int targetNode);
void pull_mode(int numVertices, bitset_t* current_frontier, bitset_t* next_frontier, 
               int* distances, bitset_t* visited, int targetNode);
// =========================================================
// 辅助函数实现
// =========================================================

/**
 * @brief 初始化 Bitset，所有位清零
 * @param bs Bitset 结构体指针
 */
void bitset_init(bitset_t *bs) {
    // 使用 memset 将整个数组清零
    memset(bs->words, 0, NUM_WORDS * sizeof(bitset_word_t));
}

/**
 * @brief 将 Bitset 的所有位设置为 1
 * @param bs Bitset 结构体指针
 */
void bitset_set_all(bitset_t *bs) {
    // 将所有存储单元设置为全 1
    memset(bs->words, 0xFF, NUM_WORDS * sizeof(bitset_word_t));
    
    // 注意：如果 BITSET_SIZE 不是 WORD_SIZE 的整数倍，
    // 最后一个 word 的多余位需要清零 (可选，取决于需求)
    size_t unused_bits = NUM_WORDS * WORD_SIZE - BITSET_SIZE;
    if (unused_bits > 0) {
        // 创建一个用于清除多余位的掩码
        bitset_word_t mask = ~((((bitset_word_t)1 << unused_bits) - 1) << (WORD_SIZE - unused_bits));
        bs->words[NUM_WORDS - 1] &= mask;
    }
}

/**
 * @brief 将 Bitset 的所有位设置为 0
 * @param bs Bitset 结构体指针
 */
void bitset_clear_all(bitset_t *bs) {
    bitset_init(bs); // 和初始化功能相同
}

/**
 * @brief 计算 Bitset 中设置为 1 的位数量（汉明重量）
 * @param bs Bitset 结构体指针
 * @return 1 的位数
 */
size_t bitset_count(const bitset_t *bs) {
    size_t count = 0;
    // 遍历每一个存储单元
    for (size_t i = 0; i < NUM_WORDS; ++i) {
        bitset_word_t word = bs->words[i];
        
        // 这是一个高效计算一个整数中 1 的位数（popcount）的循环
        // 不同的编译器和 CPU 架构可能有更快的内置函数（如 GCC 的 __builtin_popcount）
        while (word > 0) {
            word &= (word - 1); // 清除最低位的 1
            count++;
        }
    }
    return count;
}

/**
 * @brief 打印 Bitset 的内容 (仅打印 BITSET_SIZE 位)
 * @param bs Bitset 结构体指针
 */
void bitset_print(const bitset_t *bs) {
    // 从最高位打印到最低位，以匹配 std::bitset::to_string() 的阅读习惯
    for (int i = BITSET_SIZE - 1; i >= 0; --i) {
        if (BITSET_TEST(bs, i)) {
            printf("1");
        } else {
            printf("0");
        }
        // 增加空格以分隔存储单元（可选）
        if (i % WORD_SIZE == 0 && i != 0) {
            printf(" ");
        }
    }
    printf("\n");
}

int bitset_first(const bitset_t *bs) {
    if (bs == NULL) {
        return -1;
    }

    // 遍历存储位的每个 word
    for (size_t i = 0; i < NUM_WORDS; ++i) {
        bitset_word_t word = bs->words[i];

        if (word != 0) {
            // 找到了第一个非零的 word
            
            // 使用 GCC/Clang 内置函数：Count Trailing Zeros (计算末尾零的个数)
            // __builtin_ctzll 适用于 64 位 (long long)
            int bit_offset;
            
            #if defined(__GNUC__) || defined(__clang__)
                // 推荐：使用内置函数，性能最高
                bit_offset = __builtin_ctzll(word);
            #else
                // 备用方案：如果编译器不支持内置函数，需要手动实现或使用查找表
                // 这是一个非常慢的通用实现，仅用于演示
                bit_offset = 0;
                while ((word & 1) == 0 && bit_offset < WORD_SIZE) {
                    word >>= 1;
                    bit_offset++;
                }
                if (bit_offset == WORD_SIZE) {
                    // 理论上 word != 0，但为安全起见
                    continue; 
                }
            #endif

            // 计算该位在整个 bitset 中的全局索引
            // 全局索引 = (当前 word 的起始索引) + (word 内部的偏移量)
            int global_index = i * WORD_SIZE + bit_offset;

            // 注意：您可能需要检查 global_index 是否超过了总位数 (如果 bitset 未满)
            // if (global_index >= bs->total_bits) { continue; }

            return global_index;
        }
    }

    // 整个 bitset 中都没有位被设置
    return -1;
}

int bitset_next(bitset_t* bs, int prev, int numVertices) {
    for (int i = prev + 1; i < numVertices; i++) {  // 直接用參數
        if (BITSET_TEST(bs, i)) {
            return i;
        }
    }
    return -1;
}
#define CACHE_LINE_SIZE 64
#define USING_SPARSE_QUEUE 1
// --------------------------- 宏和结构体 ---------------------------

// 假设图的最大顶点数 N_MAX
#define N_MAX 100000 
// 假设图的最大边数 E_MAX
#define E_MAX 1000000

// 存储一条边的结构体
typedef struct {
    int source;
    int destination;
} Edge;

// --------------------------- 全局变量 (CSR 结果和中间数据) ---------------------------

// CSR 格式的输出数组
int ROW_PTRS[N_MAX + 1]; // 长度 N+1
int COL_INDS[E_MAX];     // 长度 E

// 中间数据
Edge edge_list[E_MAX]; // 存储从文件读取的所有边
int degrees[N_MAX] = {0}; // 存储每个顶点的出度
int max_node_id = -1; // 实际的顶点数 N = max_node_id + 1
int num_edges = 0;      // 实际的边数 E


bitset_t visited, current_frontier, next_frontier;
// ========================
// 2. 队列的数据结构定义 (用于 BFS)
// ========================

typedef struct QNode {
    int data;
    struct QNode* next;
} QNode;

#ifdef USING_SPARSE_QUEUE
typedef struct Queue {
    QNode *front; 
    char pad1[CACHE_LINE_SIZE]; // 填充，确保 front 独占一个缓存行
    QNode *rear;
    char pad2[CACHE_LINE_SIZE]; // 填充，确保 rear 独占一个缓存行
} Queue;

#else
typedef struct Queue {
    QNode *front;
    QNode *rear;
} Queue;

#endif

// ========================
// 3. 辅助函数 (创建节点, 队列操作等)
// ========================
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

// 假設 bitset 操作：BITSET_SET(bs, idx), BITSET_CLEAR(bs, idx), BITSET_TEST(bs, idx)
// BITSET_COPY(dst, src), bitset_clear_all(bs), bitset_count(bs), bitset_first(bs)
// bitset_next(bs, prev) 返回下一個 set 位 (若無，需自實作；否則用 while + first/clear)
// bitset_init(bs)

// 圖結構：CSR (鄰居列表，無向圖雙向) - ROW_PTRS[numVertices+1], COL_INDS[2*|E|]

int bfs_shortest_distance(int numVertices, int startNode, int targetNode, 
                          int* distances) {  // 只需 CSR，無 CSC
    if (startNode >= numVertices || targetNode >= numVertices || startNode < 0 || targetNode < 0) {
        fprintf(stderr, "Error: Start/Target user ID is outside the calculated graph range [0, %d].\n", numVertices - 1);
        return -1;
    }
    
    int numEdges = ROW_PTRS[numVertices];  // 總邊數 (無向圖雙向計，|E| = 2 * undirected_edges)
    if (numEdges == 0) return -1;
    
    // Push-Pull 閾值：活躍邊密度 > 0.05 切 Pull (可調)
    double threshold = 0.05;
    
    for (int i = 0; i < numVertices; i++) {
        distances[i] = -1;
    }
    distances[startNode] = 0;
    BITSET_SET(&visited, startNode);  // 假設 visited 是 bitset_t visited;

    bitset_init(&current_frontier);
    bitset_init(&next_frontier);
    BITSET_SET(&current_frontier, startNode);

    while (1) {
        if (bitset_count(&current_frontier) == 0) {
            break; 
        }
        
        // 計算活躍邊密度：當前 frontier 節點的鄰居總數 / |E|
        int active_edges = 0;
        int v = -1;
        while ((v = bitset_next(&current_frontier, v, numVertices)) != -1) {  // 迭代 frontier
            active_edges += ROW_PTRS[v + 1] - ROW_PTRS[v];
        }
        double density = (double)active_edges / numEdges;
        
        // 動態選擇模式
        if (density > threshold) {
            pull_mode(numVertices, &current_frontier, &next_frontier, distances, &visited, targetNode);
        } else {
            push_mode(numVertices, &current_frontier, &next_frontier, distances, &visited, targetNode);
        }
        
        // 檢查目標
        if (distances[targetNode] != -1 && BITSET_TEST(&visited, targetNode)) {
            return distances[targetNode];
        }
        
        // 層級推進
        BITSET_COPY(&current_frontier, &next_frontier); 
        bitset_clear_all(&next_frontier);
    }
    return -1;
}

// Push 模式（原邏輯，簡化）
void push_mode(int numVertices, bitset_t* current_frontier, bitset_t* next_frontier, 
               int* distances, bitset_t* visited, int targetNode) {
    int v = -1;
    while ((v = bitset_next(current_frontier, v, numVertices)) != -1) {
        BITSET_CLEAR(current_frontier, v);  // 移除已處理
        
        int start = ROW_PTRS[v];
        int end = ROW_PTRS[v + 1];
        for (int i = start; i < end; i++) {
            int neigh = COL_INDS[i];
            if (BITSET_TEST(visited, neigh) == 0) {
                distances[neigh] = distances[v] + 1;
                BITSET_SET(visited, neigh);
                BITSET_SET(next_frontier, neigh);
            }
        }
    }
}

// Pull 模式（使用相同鄰居列表，拉取檢查）
void pull_mode(int numVertices, bitset_t* current_frontier, bitset_t* next_frontier, 
               int* distances, bitset_t* visited, int targetNode) {
    // 優化：可只遍歷潛在 v (e.g., 所有 current_frontier 的 neigh)，但這裡簡單遍歷所有未訪問
    for (int v = 0; v < numVertices; v++) {
        if (distances[v] != -1) continue;  // 已訪問，跳過
        
        int min_dist = -1;
        int start = ROW_PTRS[v];
        int end = ROW_PTRS[v + 1];
        for (int i = start; i < end; i++) {
            int u = COL_INDS[i];  // u 是 v 的鄰居 (無向，等於入邊來源)
            if (BITSET_TEST(current_frontier, u)) {  // u 在當前層
                int candidate = distances[u] + 1;
                if (min_dist == -1 || candidate < min_dist) {
                    min_dist = candidate;
                }
            }
        }
        if (min_dist != -1) {
            distances[v] = min_dist;
            BITSET_SET(visited, v);
            BITSET_SET(next_frontier, v);
        }
    }
}

// --------------------------- 函数：读取数据文件 ---------------------------

// 从文件读取边列表，填充 edge_list 和 degrees 数组
int read_data_file(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 0; // 失败
    }

    int u, v;
    num_edges = 0;

    // 假设文件格式是 "源顶点 目标顶点"
    while (fscanf(fp, "%d %d", &u, &v) == 2) {
        if (num_edges >= E_MAX || u >= N_MAX || v >= N_MAX) {
            fprintf(stderr, "Error: Graph size exceeds limits (E_MAX or N_MAX).\n");
            fclose(fp);
            return 0;
        }

        // 记录边
        edge_list[num_edges].source = u;
        edge_list[num_edges].destination = v;
        num_edges++;

        // 统计出度 (Degree)
        degrees[u]++;
        
        // 跟踪最大的节点 ID
        if (u > max_node_id) max_node_id = u;
        if (v > max_node_id) max_node_id = v;
    }

    fclose(fp);
    return 1; // 成功
}

// --------------------------- 函数：构建 CSR ---------------------------

/**
 * 构建 CSR 格式的 ROW_PTRS 和 COL_INDS 数组。
 * * @param N 实际的顶点数 (max_node_id + 1)
 */
void build_csr(int N) {
    // --- 步骤 1 & 2: 填充 ROW_PTRS (累积和前缀和) ---
    
    // 初始化 ROW_PTRS[0]
    ROW_PTRS[0] = 0;

    // 累加度数，计算每个顶点的邻居列表的起始索引
    for (int i = 0; i < N; i++) {
        ROW_PTRS[i + 1] = ROW_PTRS[i] + degrees[i];
    }
    
    // 此时，ROW_PTRS[i] 存储的是顶点 i 的邻居在 COL_INDS 中的起始位置
    // 并且 ROW_PTRS[N] 存储了总边数 num_edges

    // --- 步骤 3: 填充 COL_INDS ---

    // 辅助数组：用于在 COL_INDS 中定位当前写入位置
    // 初始值和 ROW_PTRS 一样，但在写入 COL_INDS 时会被更新
    // 我们必须复制 ROW_PTRS，因为原数组用于查找下一个顶点的起始位置
    int *write_ptr = (int*)malloc((N) * sizeof(int));
    if (write_ptr == NULL) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }
    
    for(int i = 0; i < N; i++) {
        write_ptr[i] = ROW_PTRS[i];
    }

    // 遍历所有边
    for (int k = 0; k < num_edges; k++) {
        int u = edge_list[k].source;
        int v = edge_list[k].destination;

        // u 的邻居应该被写入到 COL_INDS[write_ptr[u]]
        int current_pos = write_ptr[u];
        
        // 写入目标顶点 v
        COL_INDS[current_pos] = v;
        
        // 更新 u 的写入指针，准备写入 u 的下一个邻居
        write_ptr[u]++;
    }

    free(write_ptr);
}

// ========================
// 5. 主程序
// ========================

int main(int argc, char *argv[]) {
    int startUser = 0; 
    int targetUser = 10;

    if (!read_data_file("facebook_combined.txt")) {
        return EXIT_FAILURE;
    }

    int N = max_node_id + 1;

    if (N > N_MAX || num_edges > E_MAX) {
        fprintf(stderr, "Error: Calculated size exceeds array limits.\n");
        return EXIT_FAILURE;
    }

    // 构建 CSR
    build_csr(N);

    // 3. 分配距离数组
    int* distances = (int*)malloc(N * sizeof(int));
    if (distances == NULL) { perror("malloc failed"); return 1; }

    // 4. 执行 BFS 查找社交距离
    printf("\n--- Shortest Distance Calculation ---\n");
    printf("Finding distance from Mapped ID %d to Mapped ID %d...\n", startUser, targetUser);
    
    int distance = bfs_shortest_distance(N, startUser, targetUser, distances);

    // 5. 输出结果
    if (distance != -1) {
        printf("\nRESULT: Social Distance (Shortest Path Length) is: %d\n", distance);
    } else {
        printf("\nRESULT: Mapped ID %d and Mapped ID %d are unreachable.\n", startUser, targetUser);
    }
    
    // 6. 清理内存
    free(distances);

    bitset_t flags;
    bitset_init(&flags); // 所有位初始化为 0

    printf("1. 初始 Bitset (全 0):\n");
    bitset_print(&flags);

    // 2. 设置第 10 位和第 63 位 (假设 WORD_SIZE 是 32 或 64)
    BITSET_SET(&flags, 10);
    BITSET_SET(&flags, 63);
    BITSET_SET(&flags, 255); // 最高位

    printf("\n2. 设置位 10, 63, 255 后:\n");
    bitset_print(&flags);

    return 0;
}