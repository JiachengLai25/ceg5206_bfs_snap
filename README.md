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
