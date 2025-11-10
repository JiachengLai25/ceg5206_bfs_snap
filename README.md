# ceg5206_bfs_snap

## 运行命令

### simple_bfs.c


```bash
cd E:\NUS\CEG5206\TeamProject\ceg5206_bfs_snap\concept_test
gcc -fopenmp -g simple_bfs.c -o simple_bfs.exe; .\simple_bfs.exe
```

### bfs_blocked_vs_baseline.c

1. 生成bfs_bench.exe文件

```bash
cd E:\NUS\CEG5206\TeamProject\ceg5206_bfs_snap\concept_test
gcc -O2 -fopenmp bfs_blocked_vs_baseline.c simple_bfs.c -o bfs_bench.exe -DNO_MAIN
```

2. 生成bfs_benchmark.csv文件

```bash
.\bfs_bench.exe facebook_combined.txt 0 2000
```

3. py绘图

```bash
python plot_bfs_benchmark.py
```

### cache_benchmark.py

作用：对比Cache miss/bandwidth comparison for CSR vs. Blocked-CSR. (需要在linux环境下运行，此处使用wsl)

1. 进入 wsl 并安装 perf

```bash
wsl
sudo apt update
sudo apt install linux-tools-common linux-tools-generic -y
```

2. 生成bfs_bench.out

```bash
gcc -O2 -fopenmp bfs_blocked_vs_baseline.c bitset.c -o bfs_bench.out -DNO_MAIN
```

3. 使用bfs_bench.out生成bfs_benchmark.csv文件

```bash
./bfs_bench.out facebook_processed.txt 0 2000
```

4. 执行 cache_benchmark.py 进行批量性能采样，生成cpu_benchmark_wsl.csv

```bash
python3 cache_benchmark_wsl.py
```

5. 创建虚拟环境并安装绘图包

6. 使用 plot_cpu_benchmark.py 绘图

```bash
python3 plot_cpu_benchmark.py
```

