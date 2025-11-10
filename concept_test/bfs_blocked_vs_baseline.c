// bfs_blocked_vs_baseline_final.c
// Build: gcc -O2 -fopenmp bfs_blocked_vs_baseline_final.c simple_bfs.c -o bfs_bench_final.exe -DNO_MAIN
// Run  : .\bfs_bench_final.exe facebook_combined.txt 0 2000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "bitset.h"

// --------------------------- CONFIG ---------------------------
#define BLOCK_SIZE 32         // 更细粒度，提升块覆盖率
#define THRESHOLD  0.05       // push/pull 阈值
#define REPEAT_RUN 5          // 每个配置重复次数取平均

// --------------------------- CSR ---------------------------
typedef struct { int n, m; int *rowptr, *colind; } CSR;

// --------------------------- BCSR（邻接仍用 CSR） ---------------------------
typedef struct {
    int n, m, num_blocks;
    int *block_offsets;            // 每块首顶点
    const int *rowptr, *colind;    // 引用 CSR
} BCSR;

// --------------------------- bitset 辅助（不改头文件） ---------------------------
static int bitset_next_local(const bitset_t *bs, int prev, int limit){
    int i = prev < 0 ? 0 : prev+1;
    for (; i < limit && i < BITSET_SIZE; ++i)
        if (BITSET_TEST(bs, i)) return i;
    return -1;
}

// --------------------------- 读边 & 构建 CSR ---------------------------
static int read_edge_list(const char* fn, int **S, int **T, int *m, int *n){
    FILE *fp = fopen(fn, "r"); if(!fp){perror("open"); return 0;}
    int cap=1<<20, u,v; *S=malloc(4*cap); *T=malloc(4*cap);
    int M=0, N=-1;
    while (fscanf(fp, "%d %d", &u,&v)==2){
        if (M>=cap){ cap<<=1; *S=realloc(*S,4*cap); *T=realloc(*T,4*cap); }
        (*S)[M]=u; (*T)[M]=v; if(u>N)N=u; if(v>N)N=v; M++;
    }
    fclose(fp); *m=M; *n=N+1; return 1;
}
static int build_csr(CSR* G, int n, int m, const int* S, const int* T){
    G->n=n; G->m=m; G->rowptr=calloc(n+1,4); G->colind=malloc(4*m);
    for(int i=0;i<m;i++) G->rowptr[S[i]]++;
    for(int i=0,acc=0;i<n;i++){ int d=G->rowptr[i]; G->rowptr[i]=acc; acc+=d; }
    G->rowptr[n]=m;
    int *wp=malloc(4*n); memcpy(wp,G->rowptr,4*n);
    for(int i=0;i<m;i++){ int u=S[i]; G->colind[wp[u]++]=T[i]; }
    free(wp); return 1;
}

// --------------------------- 构建 BCSR ---------------------------
static BCSR* build_bcsr(const CSR* G){
    BCSR* B = malloc(sizeof(BCSR));
    B->n=G->n; B->m=G->m; B->rowptr=G->rowptr; B->colind=G->colind;
    B->num_blocks = (G->n + BLOCK_SIZE - 1)/BLOCK_SIZE;
    B->block_offsets = malloc(4*B->num_blocks);
    for(int b=0;b<B->num_blocks;b++) B->block_offsets[b]=b*BLOCK_SIZE;
    return B;
}
static void free_bcsr(BCSR* B){ if(!B) return; free(B->block_offsets); free(B); }

// --------------------------- frontier 密度 ---------------------------
static double frontier_density_edges(const CSR* G, const bitset_t* f){
    long long active=0; int v=-1;
    while((v=bitset_next_local(f,v,G->n))!=-1)
        active += (G->rowptr[v+1]-G->rowptr[v]);
    return (G->m==0)?0.0:(double)active/G->m;
}

// --------------------------- BFS (CSR) ---------------------------
static int bfs_csr(const CSR* G, int s, int t){
    int n=G->n; int *dist=malloc(4*n);
    for(int i=0;i<n;i++) dist[i]=-1;
    bitset_t vis, cur, nxt; bitset_init(&vis); bitset_init(&cur); bitset_init(&nxt);
    dist[s]=0; BITSET_SET(&vis,s); BITSET_SET(&cur,s);

    while (1){
        if (bitset_count(&cur)==0) break;      // 无节点可扩展
        double dens = frontier_density_edges(G, &cur);

        if (dens < THRESHOLD){ // PUSH
            #pragma omp parallel for schedule(dynamic,64)
            for(int v=0; v<n; v++){
                if(!BITSET_TEST(&cur,v)) continue;
                int s0=G->rowptr[v], e0=G->rowptr[v+1];
                for(int i=s0;i<e0;i++){
                    int u=G->colind[i];
                    if(!BITSET_TEST(&vis,u)){
                        #pragma omp critical
                        {
                            if(!BITSET_TEST(&vis,u)){
                                dist[u]=dist[v]+1;
                                BITSET_SET(&vis,u);
                                BITSET_SET(&nxt,u);
                            }
                        }
                    }
                }
            }
        }else{                // PULL
            #pragma omp parallel for schedule(dynamic,256)
            for(int v=0; v<n; v++){
                if(dist[v]!=-1) continue;
                int s0=G->rowptr[v], e0=G->rowptr[v+1], nd=-1;
                for(int i=s0;i<e0;i++){
                    int u=G->colind[i];
                    if(BITSET_TEST(&cur,u)){ nd = dist[u]+1; break; }
                }
                if (nd!=-1){
                    #pragma omp critical
                    {
                        if(dist[v]==-1){
                            dist[v]=nd;
                            BITSET_SET(&vis,v);
                            BITSET_SET(&nxt,v);
                        }
                    }
                }
            }
        }

        // 层推进：先交换再检查，避免“过早退出 + 读到空集”
        BITSET_COPY(&cur,&nxt);
        bitset_clear_all(&nxt);
        if (dist[t]!=-1) break;
    }
    int ans=dist[t]; free(dist); return ans;
}

// --------------------------- BFS (BCSR) ---------------------------
static int bfs_bcsr(const BCSR* B, int s, int t){
    int n=B->n; const CSR G = {B->n,B->m,(int*)B->rowptr,(int*)B->colind};
    int *dist=malloc(4*n);
    for(int i=0;i<n;i++) dist[i]=-1;
    bitset_t vis, cur, nxt; bitset_init(&vis); bitset_init(&cur); bitset_init(&nxt);
    dist[s]=0; BITSET_SET(&vis,s); BITSET_SET(&cur,s);

    while (1){
        if (bitset_count(&cur)==0) break;
        double dens = frontier_density_edges(&G, &cur);

        if (dens < THRESHOLD){ // PUSH：按块划分
            #pragma omp parallel
            {
                int T=omp_get_num_threads(), tid=omp_get_thread_num();
                int blk_per = (B->num_blocks + T - 1)/T;
                int b0 = tid*blk_per, b1 = b0+blk_per; if (b1>B->num_blocks) b1=B->num_blocks;

                for(int b=b0; b<b1; b++){
                    int vfirst = B->block_offsets[b];
                    int vlast  = (b==B->num_blocks-1)? (n-1):(vfirst+BLOCK_SIZE-1);
                    for(int v=vfirst; v<=vlast && v<n; v++){
                        if(!BITSET_TEST(&cur,v)) continue;
                        int s0=B->rowptr[v], e0=B->rowptr[v+1];
                        for(int i=s0;i<e0;i++){
                            int u=B->colind[i];
                            if(!BITSET_TEST(&vis,u)){
                                #pragma omp critical
                                {
                                    if(!BITSET_TEST(&vis,u)){
                                        dist[u]=dist[v]+1;
                                        BITSET_SET(&vis,u);
                                        BITSET_SET(&nxt,u);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }else{                 // PULL：按块遍历未访问顶点
            #pragma omp parallel for schedule(dynamic)
            for(int b=0; b<B->num_blocks; b++){
                int vfirst = B->block_offsets[b];
                int vlast  = (b==B->num_blocks-1)? (n-1):(vfirst+BLOCK_SIZE-1);
                for(int v=vfirst; v<=vlast && v<n; v++){
                    if(dist[v]!=-1) continue;
                    int s0=B->rowptr[v], e0=B->rowptr[v+1], nd=-1;
                    for(int i=s0;i<e0;i++){
                        int u=B->colind[i];
                        if(BITSET_TEST(&cur,u)){ nd=dist[u]+1; break; }
                    }
                    if (nd!=-1){
                        #pragma omp critical
                        {
                            if(dist[v]==-1){
                                dist[v]=nd;
                                BITSET_SET(&vis,v);
                                BITSET_SET(&nxt,v);
                            }
                        }
                    }
                }
            }
        }

        BITSET_COPY(&cur,&nxt);
        bitset_clear_all(&nxt);
        if (dist[t]!=-1) break;
    }
    int ans=dist[t]; free(dist); return ans;
}

// --------------------------- 计时封装（多次取平均） ---------------------------
typedef struct { int dist; double teps; double ms; } RunStat;

static RunStat run_avg_csr(const CSR* G, int s, int t){
    double total=0; int d=-1;
    for(int r=0;r<REPEAT_RUN;r++){ double t0=omp_get_wtime(); d=bfs_csr(G,s,t); total += omp_get_wtime()-t0; }
    double sec=total/REPEAT_RUN;
    RunStat R={d,(sec>0? G->m/sec:0.0),sec*1000.0};
    return R;
}
static RunStat run_avg_bcsr(const BCSR* B, int s, int t){
    double total=0; int d=-1;
    for(int r=0;r<REPEAT_RUN;r++){ double t0=omp_get_wtime(); d=bfs_bcsr(B,s,t); total += omp_get_wtime()-t0; }
    double sec=total/REPEAT_RUN;
    RunStat R={d,(sec>0? B->m/sec:0.0),sec*1000.0};
    return R;
}

// --------------------------- 主程序 ---------------------------
int main(int argc,char**argv){
    if(argc<4){ printf("Usage: %s <edge_list> <src> <dst>\n",argv[0]); return 1; }
    const char* fn=argv[1]; int src=atoi(argv[2]), dst=atoi(argv[3]);

    int *S=NULL,*T=NULL,m=0,n=0;
    if(!read_edge_list(fn,&S,&T,&m,&n)){ return 2; }
    printf("Read edges: V=%d, E=%d\n", n, m);

    CSR G={0}; build_csr(&G,n,m,S,T); free(S); free(T);
    BCSR *B = build_bcsr(&G);

    FILE *csv=fopen("bfs_benchmark.csv","w");
    fprintf(csv,"impl,threads,dist,teps,time_ms\n");

    int ths[]={1,2,4,8,16,32};
    int thn=sizeof(ths)/sizeof(ths[0]);

    for(int i=0;i<thn;i++){
        int T=ths[i]; omp_set_num_threads(T);

        RunStat r1 = run_avg_csr(&G,src,dst);
        RunStat r2 = run_avg_bcsr(B,src,dst);

        // 一致性校验
        if (r1.dist != r2.dist)
            printf("⚠️ Dist mismatch: CSR=%d, BCSR=%d\n", r1.dist, r2.dist);

        printf("[CSR ] threads=%d dist=%d TEPS=%.2f time=%.3fms\n", T, r1.dist, r1.teps, r1.ms);
        printf("[BCSR] threads=%d dist=%d TEPS=%.2f time=%.3fms\n", T, r2.dist, r2.teps, r2.ms);

        fprintf(csv,"CSR,%d,%d,%.6f,%.3f\n",  T, r1.dist, r1.teps, r1.ms);
        fprintf(csv,"BCSR,%d,%d,%.6f,%.3f\n", T, r2.dist, r2.teps, r2.ms);
        fflush(csv);
    }
    fclose(csv);
    printf("✅ CSV written: bfs_benchmark.csv\n");

    free_bcsr(B); free(G.rowptr); free(G.colind);
    return 0;
}
