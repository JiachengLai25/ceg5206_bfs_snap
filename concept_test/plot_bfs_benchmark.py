import csv
import matplotlib.pyplot as plt

threads = []
csr_teps = {}
bcsr_teps = {}

with open('bfs_benchmark.csv', 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        impl = row['impl']
        th   = int(row['threads'])
        teps = float(row['teps'])
        if impl == 'CSR':
            csr_teps[th] = teps
        elif impl == 'BCSR':
            bcsr_teps[th] = teps

xs = sorted(set(list(csr_teps.keys()) + list(bcsr_teps.keys())))
y1 = [csr_teps.get(x, None) for x in xs]
y2 = [bcsr_teps.get(x, None) for x in xs]

plt.figure()
plt.plot(xs, y1, marker='o', label='CSR (baseline)')
plt.plot(xs, y2, marker='o', label='Blocked-CSR')
plt.xlabel('Threads')
plt.ylabel('TEPS')
plt.title('BFS Throughput vs Threads')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bfs_benchmark.png', dpi=150)
print('✅ Saved: bfs_benchmark.png')
