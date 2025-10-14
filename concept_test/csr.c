#include <stdio.h>
#include <stdlib.h>

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

// --------------------------- 主函数：测试 ---------------------------

int main() {
    // 确保你的数据文件 'data_file.txt' 在程序目录下
    if (!read_data_file("facebook_combined.txt")) {
        return EXIT_FAILURE;
    }
    
    // N 是实际的顶点数 (例如：如果最大 ID 是 3，则 N=4 (0, 1, 2, 3))
    int N = max_node_id + 1;

    printf("--- 读入统计 ---\n");
    printf("顶点数 (N): %d\n", N);
    printf("边数 (E): %d\n", num_edges);
    
    // 确保有足够的空间
    if (N > N_MAX || num_edges > E_MAX) {
        fprintf(stderr, "Error: Calculated size exceeds array limits.\n");
        return EXIT_FAILURE;
    }

    // 构建 CSR
    build_csr(N);

    // --- 打印 CSR 结果 ---
    printf("\n--- CSR 结果 ---\n");
    
    printf("ROW_PTRS (长度 %d): ", N + 1);
    for (int i = 0; i <= N; i++) {
        printf("%d ", ROW_PTRS[i]);
    }
    printf("\n");

    printf("COL_INDS (长度 %d): ", num_edges);
    for (int i = 0; i < num_edges; i++) {
        printf("%d ", COL_INDS[i]);
    }
    printf("\n");
    
    // --- 演示如何使用 CSR 查找邻居 ---
    printf("\n--- 查找邻居演示 ---\n");
    for (int u = 0; u < N; u++) {
        int start = ROW_PTRS[u];
        int end = ROW_PTRS[u + 1];
        
        printf("顶点 %d 的邻居: [", u);
        for (int i = start; i < end; i++) {
            printf("%d ", COL_INDS[i]);
        }
        printf("]\n");
    }

    return EXIT_SUCCESS;
}