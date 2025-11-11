import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process

# Detect system NUMA node configuration
def get_numa_nodes():
    """Get information about NUMA nodes in the system"""
    try:
        # Read nodes under /sys/devices/system/node
        nodes = [node for node in os.listdir('/sys/devices/system/node') 
                 if node.startswith('node')]
        num_nodes = len(nodes)
        if num_nodes < 2:
            print("Warning: The system has fewer than 2 NUMA nodes, making it difficult to test NUMA placement effects effectively.")
        return num_nodes
    except Exception as e:
        print(f"Failed to get NUMA node information: {e}")
        return 0

# Memory-intensive test task (simulating memory access patterns in graph algorithms)
def memory_intensive_task(size_mb=1024, iterations=100):
    """Memory-intensive task: continuous read/write of large memory blocks (simulating CSR structure access)"""
    # Allocate memory block (approximately size_mb MB)
    data = np.random.rand(int(size_mb * 1024 * 1024 / 8))  # float64 occupies 8 bytes
    start_time = time.time()
    
    # Simulate random access pattern (similar to adjacency list traversal in graphs)
    for _ in range(iterations):
        indices = np.random.randint(0, len(data), size=int(len(data)*0.1))
        data[indices] += 1.0  # Randomly update part of the memory
    
    elapsed = time.time() - start_time
    return elapsed

# Run task with specified NUMA policy using numactl
def run_with_numa_policy(node_cpu, node_mem, task_func, *args):
    """
    Execute a task with a specified NUMA CPU and memory binding policy using numactl.
    
    Parameters:
        node_cpu: CPU node to bind to
        node_mem: Memory node to bind to
        task_func: Task function to execute
        *args: Arguments for the task function
    """
    if not shutil.which('numactl'):
        raise RuntimeError("numactl tool not found. Please install numactl to test NUMA policies.")
    
    # Construct numactl command
    cmd = [
        'numactl',
        f'--cpunodebind={node_cpu}',
        f'--membind={node_mem}',
        'python3', '-c',
        f'import numpy as np; from __main__ import {task_func.__name__}; '
        f'print({task_func.__name__}(*{args}))'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command execution failed: {result.stderr}")
        return None
    return float(result.stdout.strip())

# Multi-process testing of different NUMA placement strategies
def test_numa_placement(num_nodes, size_mb=1024, iterations=100, repeats=5):
    """Test the performance of different NUMA placement strategies"""
    strategies = [
        ("Local Placement (CPU=0, MEM=0)", 0, 0),
        ("Local Placement (CPU=1, MEM=1)", 1, 1),
        ("Remote Placement (CPU=0, MEM=1)", 0, 1),
        ("Remote Placement (CPU=1, MEM=0)", 1, 0),
        ("Interleaved Placement (CPU=0-1, MEM=0-1)", f"0-{num_nodes-1}", f"0-{num_nodes-1}")
    ]
    
    # Store results
    results = {name: [] for name, _, _ in strategies}
    
    # Repeat tests multiple times and take the average
    for _ in range(repeats):
        for name, cpu_node, mem_node in strategies:
            try:
                elapsed = run_with_numa_policy(cpu_node, mem_node, 
                                              memory_intensive_task, 
                                              size_mb, iterations)
                if elapsed:
                    results[name].append(elapsed)
                    print(f"{name} - Time elapsed: {elapsed:.2f} seconds")
            except Exception as e:
                print(f"{name} test failed: {e}")
    
    # Compute average time
    avg_results = {name: np.mean(times) for name, times in results.items() if times}
    return avg_results

# Visualize NUMA placement performance comparison
def plot_numa_comparison(results):
    """Plot performance comparison across different NUMA placement strategies"""
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    times = list(results.values())
    
    # Compute relative performance (based on best performance)
    best_time = min(times)
    rel_performance = [best_time / t * 100 for t in times]  # Relative performance in percentage
    
    x = np.arange(len(names))
    bars = plt.bar(x, rel_performance, color=['#4CAF50', '#4CAF50', '#F44336', '#F44336', '#FFC107'])
    
    # Add data labels
    for bar, rel, t in zip(bars, rel_performance, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{rel:.1f}%\n({t:.2f}s)",
                ha='center', va='bottom', rotation=0)
    
    plt.xlabel('NUMA Placement Strategy')
    plt.ylabel('Relative Performance (%)')
    plt.title('Impact of NUMA Placement Strategy on Memory-Intensive Task Performance')
    plt.xticks(x, names, rotation=15, ha='right')
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return plt

# Main function
if __name__ == "__main__":
    import shutil
    
    # Check NUMA nodes
    num_numa_nodes = get_numa_nodes()
    if num_numa_nodes < 2:
        print("Exiting test: At least 2 NUMA nodes are required.")
        exit(1)
    
    # Execute test (adjust parameters as needed)
    print(f"Starting NUMA placement performance test, node count: {num_numa_nodes}")
    numa_results = test_numa_placement(
        num_nodes=num_numa_nodes,
        size_mb=2048,  # 2GB memory block
        iterations=50,
        repeats=3
    )
    
    # Plot results
    if numa_results:
        plt = plot_numa_comparison(numa_results)
        plt.show()
    else:
        print("No valid test results obtained.")