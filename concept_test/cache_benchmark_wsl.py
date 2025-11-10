import subprocess
import csv

# ==============================
# ✅ 基本配置
# ==============================
binary = "./bfs_bench.out"
graph_file = "test_graph.txt"
threads_list = [1, 2, 4, 8, 16, 32]
impls = ["CSR", "BCSR"]

# 要采集的 perf 事件
events = [
    "cycles",
    "instructions",
    "branches",
    "branch-misses"
]

output_csv = "cpu_benchmark_wsl.csv"


# ==============================
# ✅ 运行 perf 并返回输出
# ==============================
def run_perf(impl, threads):
    cmd = [
        "perf", "stat", "-x,", "-e", ",".join(events),
        binary, graph_file, "0", "5", "--impl=" + impl, "--threads=" + str(threads)
    ]
    print(f"\n▶ Running {impl} with {threads} threads...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr.strip() == "":
        print("⚠️ No perf output detected (check permissions or path).")
    return result.stderr


# ==============================
# ✅ 解析 perf 输出
# ==============================
def parse_perf_output(perf_output):
    metrics = {}
    for line in perf_output.splitlines():
        if not line.strip():
            continue
        if line.startswith("Performance") or line.startswith("#") or "error" in line or "Unable" in line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                value = float(parts[0])
                event = parts[2]  # 第三列是事件名，如 cycles:u
                event = event.split(":")[0]  # 去掉 ":u"
                metrics[event] = value
            except ValueError:
                continue

    # 派生指标计算
    cycles = metrics.get("cycles", 0)
    instr = metrics.get("instructions", 0)
    branches = metrics.get("branches", 0)
    misses = metrics.get("branch-misses", 0)

    metrics["IPC"] = instr / cycles if cycles > 0 else 0
    metrics["Branch_Miss_Rate"] = misses / branches if branches > 0 else 0

    return metrics


# ==============================
# ✅ 主流程：循环采样并写入 CSV
# ==============================
def main():
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "impl", "threads",
            "instructions", "cycles", "IPC",
            "branches", "branch-misses", "Branch_Miss_Rate"
        ])

        for impl in impls:
            for t in threads_list:
                perf_out = run_perf(impl, t)
                metrics = parse_perf_output(perf_out)

                writer.writerow([
                    impl, t,
                    int(metrics.get("instructions", 0)),
                    int(metrics.get("cycles", 0)),
                    round(metrics.get("IPC", 4), 4),
                    int(metrics.get("branches", 0)),
                    int(metrics.get("branch-misses", 0)),
                    round(metrics.get("Branch_Miss_Rate", 6), 6)
                ])
    print(f"\n✅ Benchmark results saved to {output_csv}")


if __name__ == "__main__":
    main()
