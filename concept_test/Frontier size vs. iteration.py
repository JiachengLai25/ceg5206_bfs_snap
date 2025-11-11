import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import pandas as pd

def simulate_bfs_frontier_trajectory():
    """
    模拟BFS算法中Push和Pull模式下的Frontier size变化轨迹
    基于真实的BFS算法行为进行模拟
    """
    # 设置随机种子以保证结果可重现
    np.random.seed(42)
    
    # 模拟参数
    total_nodes = 10000
    max_iterations = 20
    density_threshold = 0.05  # Push/Pull切换阈值
    
    # 初始化数据存储
    iterations = []
    frontier_sizes = []
    modes = []
    densities = []
    
    # 模拟BFS过程
    current_frontier_size = 1  # 起始节点
    iteration = 0
    
    while current_frontier_size > 0 and iteration < max_iterations:
        # 计算当前前沿的边密度（模拟）
        if iteration == 0:
            current_density = 0.001  # 初始很低
        else:
            # 密度随迭代变化：先增长后下降
            peak_iteration = max_iterations // 3
            if iteration < peak_iteration:
                current_density = min(0.15, 0.001 * (2 ** iteration))
            else:
                current_density = max(0.001, 0.15 * (0.7 ** (iteration - peak_iteration)))
        
        # 根据密度决定模式
        if current_density < density_threshold:
            mode = "Push"
            # Push模式下前沿增长较慢（处理稀疏图）
            if iteration < 5:
                growth_factor = np.random.normal(2.5, 0.3)
            else:
                growth_factor = np.random.normal(1.8, 0.2)
        else:
            mode = "Pull" 
            # Pull模式下前沿可能收缩（处理稠密图）
            growth_factor = np.random.normal(0.7, 0.15)
        
        # 添加噪声和随机性
        noise = np.random.normal(1, 0.1)
        growth_factor *= noise
        
        # 计算新的前沿大小（限制在合理范围内）
        new_frontier_size = int(current_frontier_size * growth_factor)
        new_frontier_size = max(1, min(total_nodes * 0.8, new_frontier_size))
        
        # 记录数据
        iterations.append(iteration)
        frontier_sizes.append(current_frontier_size)
        modes.append(mode)
        densities.append(current_density)
        
        # 为下一次迭代更新
        current_frontier_size = new_frontier_size
        iteration += 1
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Iteration': iterations,
        'Frontier_Size': frontier_sizes,
        'Mode': modes,
        'Density': densities
    })
    
    return df

def plot_frontier_trajectory(df):
    """
    绘制Frontier size随迭代变化的轨迹图
    """
    # 设置绘图风格
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 颜色映射
    colors = {'Push': '#2E86AB', 'Pull': '#A23B72'}
    
    # 主图：Frontier size轨迹
    for mode in ['Push', 'Pull']:
        mask = df['Mode'] == mode
        ax1.plot(df[mask]['Iteration'], df[mask]['Frontier_Size'], 
                marker='o', linewidth=3, markersize=8, label=mode,
                color=colors[mode], alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frontier Size', fontsize=14, fontweight='bold')
    ax1.set_title('BFS Frontier Size vs. Iteration\n(Push vs. Pull Mode Trajectory)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_yscale('log')  # 对数尺度更好地显示变化
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # 添加模式切换的标注
    for i in range(1, len(df)):
        if df['Mode'].iloc[i] != df['Mode'].iloc[i-1]:
            ax1.annotate(f'{df["Mode"].iloc[i]} Mode', 
                        xy=(df['Iteration'].iloc[i], df['Frontier_Size'].iloc[i]),
                        xytext=(10, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, fontweight='bold', color='red')
    
    # 子图：密度和模式关系
    ax2b = ax2.twinx()
    
    # 密度曲线
    ax2.plot(df['Iteration'], df['Density'], color='#F18F01', 
            linewidth=2, marker='s', markersize=6, label='Frontier Density')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
               label='Density Threshold (0.05)')
    
    # 模式背景色
    for i in range(len(df) - 1):
        mode = df['Mode'].iloc[i]
        ax2b.axvspan(df['Iteration'].iloc[i], df['Iteration'].iloc[i+1], 
                    alpha=0.2, color=colors[mode])
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frontier Density', fontsize=12, fontweight='bold', color='#F18F01')
    ax2.set_title('Frontier Density and Mode Switching', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 创建图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    patches = [Patch(color=colors['Push'], alpha=0.3, label='Push Mode'),
              Patch(color=colors['Pull'], alpha=0.3, label='Pull Mode')]
    ax2.legend(lines1 + patches, labels1 + ['Push Mode', 'Pull Mode'], 
              loc='upper right')
    
    plt.tight_layout()
    return fig

def create_comparison_analysis(df):
    """
    创建统计比较分析
    """
    # 模式统计
    mode_stats = df.groupby('Mode').agg({
        'Frontier_Size': ['mean', 'std', 'max'],
        'Iteration': 'count'
    }).round(2)
    
    # 性能指标计算
    total_iterations = len(df)
    push_iterations = len(df[df['Mode'] == 'Push'])
    pull_iterations = len(df[df['Mode'] == 'Pull'])
    
    print("=== BFS Frontier轨迹分析 ===")
    print(f"总迭代次数: {total_iterations}")
    print(f"Push模式迭代: {push_iterations} ({push_iterations/total_iterations*100:.1f}%)")
    print(f"Pull模式迭代: {pull_iterations} ({pull_iterations/total_iterations*100:.1f}%)")
    print(f"模式切换次数: {len(df[df['Mode'] != df['Mode'].shift()]) - 1}")
    print("\n各模式统计:")
    print(mode_stats)
    
    return mode_stats

def create_animated_frontier_plot(df):
    """
    创建动态前沿变化图（可选）
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        current_data = df[df['Iteration'] <= frame]
        
        colors = {'Push': '#2E86AB', 'Pull': '#A23B72'}
        
        for mode in ['Push', 'Pull']:
            mask = current_data['Mode'] == mode
            if len(current_data[mask]) > 0:
                ax.plot(current_data[mask]['Iteration'], 
                       current_data[mask]['Frontier_Size'], 
                       marker='o', linewidth=2, label=mode, color=colors[mode])
        
        ax.set_xlim(0, len(df))
        ax.set_ylim(1, df['Frontier_Size'].max() * 1.1)
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Frontier Size (log scale)')
        ax.set_title(f'BFS Frontier Evolution (Iteration {frame})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    # 注释掉动画创建以避免在静态环境中运行
    # anim = FuncAnimation(fig, update, frames=len(df), interval=500, repeat=False)
    plt.close()
    
    return "动画功能已准备（在实际环境中取消注释相关代码）"

# 主执行函数
def main():
    """主执行函数"""
    print("生成BFS Frontier轨迹数据...")
    
    # 生成模拟数据
    df = simulate_bfs_frontier_trajectory()
    
    # 绘制轨迹图
    print("绘制轨迹图...")
    fig = plot_frontier_trajectory(df)
    
    # 统计分析
    stats = create_comparison_analysis(df)
    
    # 保存图表
    plt.savefig('bfs_frontier_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig('bfs_frontier_trajectory.pdf', bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    # 保存数据
    df.to_csv('bfs_frontier_data.csv', index=False)
    
    print("\n图表已保存为 'bfs_frontier_trajectory.png'")
    print("数据已保存为 'bfs_frontier_data.csv'")
    
    return df, fig, stats

# 可选：创建不同参数下的对比图
def create_parameter_comparison():
    """创建不同参数下的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    density_thresholds = [0.01, 0.05, 0.1, 0.2]
    
    for i, threshold in enumerate(density_thresholds):
        np.random.seed(42)  # 保持其他参数一致
        df = simulate_bfs_frontier_trajectory()
        
        colors = {'Push': '#2E86AB', 'Pull': '#A23B72'}
        
        for mode in ['Push', 'Pull']:
            mask = df['Mode'] == mode
            axes[i].plot(df[mask]['Iteration'], df[mask]['Frontier_Size'], 
                       marker='o', linewidth=2, label=mode, color=colors[mode])
        
        axes[i].set_yscale('log')
        axes[i].set_title(f'Density Threshold = {threshold}', fontweight='bold')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Frontier Size')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.suptitle('BFS Frontier Trajectory under Different Density Thresholds', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('bfs_threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 运行主程序
    df, fig, stats = main()
    
    # 可选：运行参数比较
    # create_parameter_comparison()