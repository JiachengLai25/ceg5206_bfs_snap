import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 读取 CSV 文件
# ===============================
df = pd.read_csv("cpu_benchmark_wsl.csv")
df = df.sort_values(by=["impl", "threads"])

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 12

# ===============================
# 创建 3 个子图（纵向排列）
# ===============================
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
metrics = [
    ("IPC", "IPC (Instructions per Cycle)", "↑ Higher = Better"),
    ("Branch_Miss_Rate", "Branch Miss Rate", "↓ Lower = Better"),
    ("cycles", "Total Cycles", "↓ Lower = Faster")
]

colors = {"CSR": "#1f77b4", "BCSR": "#ff7f0e"}

for i, (metric, ylabel, note) in enumerate(metrics):
    ax = axes[i]
    for impl in ["CSR", "BCSR"]:
        sub = df[df["impl"] == impl]
        ax.plot(sub["threads"], sub[metric], marker="o", label=impl, color=colors[impl])
    ax.set_ylabel(f"{ylabel}\n({note})", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    if i == 0:
        ax.set_title("BFS Performance Comparison: CSR vs Blocked-CSR", fontsize=14, pad=10)
    if i == len(metrics) - 1:
        ax.set_xlabel("Threads", fontsize=12)

axes[0].legend(loc="best", frameon=True)

plt.tight_layout()
plt.savefig("bfs_perf_summary.png", dpi=200)
plt.close()

print("✅ Combined plot saved as bfs_perf_summary.png")
