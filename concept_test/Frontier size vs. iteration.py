import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import pandas as pd

def simulate_bfs_frontier_trajectory():
    """
    Simulate the trajectory of Frontier size variation under Push and Pull modes in the BFS algorithm.
    This simulation is based on realistic BFS algorithm behavior.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulation parameters
    total_nodes = 10000
    max_iterations = 20
    density_threshold = 0.05  # Threshold for switching between Push and Pull modes
    
    # Initialize data storage
    iterations = []
    frontier_sizes = []
    modes = []
    densities = []
    
    # Simulate BFS process
    current_frontier_size = 1  # Starting node
    iteration = 0
    
    while current_frontier_size > 0 and iteration < max_iterations:
        # Compute the current frontier edge density (simulated)
        if iteration == 0:
            current_density = 0.001  # Very low initially
        else:
            # Density increases first and then decreases with iteration
            peak_iteration = max_iterations // 3
            if iteration < peak_iteration:
                current_density = min(0.15, 0.001 * (2 ** iteration))
            else:
                current_density = max(0.001, 0.15 * (0.7 ** (iteration - peak_iteration)))
        
        # Determine mode based on density
        if current_density < density_threshold:
            mode = "Push"
            # In Push mode, frontier grows more slowly (for sparse graphs)
            if iteration < 5:
                growth_factor = np.random.normal(2.5, 0.3)
            else:
                growth_factor = np.random.normal(1.8, 0.2)
        else:
            mode = "Pull"
            # In Pull mode, frontier may shrink (for dense graphs)
            growth_factor = np.random.normal(0.7, 0.15)
        
        # Add noise and randomness
        noise = np.random.normal(1, 0.1)
        growth_factor *= noise
        
        # Compute new frontier size (bounded within a reasonable range)
        new_frontier_size = int(current_frontier_size * growth_factor)
        new_frontier_size = max(1, min(total_nodes * 0.8, new_frontier_size))
        
        # Record data
        iterations.append(iteration)
        frontier_sizes.append(current_frontier_size)
        modes.append(mode)
        densities.append(current_density)
        
        # Update for next iteration
        current_frontier_size = new_frontier_size
        iteration += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Iteration': iterations,
        'Frontier_Size': frontier_sizes,
        'Mode': modes,
        'Density': densities
    })
    
    return df

def plot_frontier_trajectory(df):
    """
    Plot the trajectory of Frontier size variation over iterations.
    """
    # Set plot style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Color mapping
    colors = {'Push': '#2E86AB', 'Pull': '#A23B72'}
    
    # Main plot: Frontier size trajectory
    for mode in ['Push', 'Pull']:
        mask = df['Mode'] == mode
        ax1.plot(df[mask]['Iteration'], df[mask]['Frontier_Size'], 
                marker='o', linewidth=3, markersize=8, label=mode,
                color=colors[mode], alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frontier Size', fontsize=14, fontweight='bold')
    ax1.set_title('BFS Frontier Size vs. Iteration\n(Push vs. Pull Mode Trajectory)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_yscale('log')  # Log scale for better visibility
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Annotate mode switching
    for i in range(1, len(df)):
        if df['Mode'].iloc[i] != df['Mode'].iloc[i-1]:
            ax1.annotate(f'{df["Mode"].iloc[i]} Mode', 
                        xy=(df['Iteration'].iloc[i], df['Frontier_Size'].iloc[i]),
                        xytext=(10, 30), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, fontweight='bold', color='red')
    
    # Subplot: Density and mode relationship
    ax2b = ax2.twinx()
    
    # Density curve
    ax2.plot(df['Iteration'], df['Density'], color='#F18F01', 
            linewidth=2, marker='s', markersize=6, label='Frontier Density')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
               label='Density Threshold (0.05)')
    
    # Mode background shading
    for i in range(len(df) - 1):
        mode = df['Mode'].iloc[i]
        ax2b.axvspan(df['Iteration'].iloc[i], df['Iteration'].iloc[i+1], 
                    alpha=0.2, color=colors[mode])
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frontier Density', fontsize=12, fontweight='bold', color='#F18F01')
    ax2.set_title('Frontier Density and Mode Switching', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Create legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    patches = [Patch(color=colors['Push'], alpha=0.3, label='Push Mode'),
              Patch(color=colors['Pull'], alpha=0.3, label='Pull Mode')]
    ax2.legend(lines1 + patches, labels1 + ['Push Mode', 'Pull Mode'], 
              loc='upper right')
    
    plt.tight_layout()
    return fig

def create_comparison_analysis(df):
    """
    Create statistical comparison analysis.
    """
    # Mode statistics
    mode_stats = df.groupby('Mode').agg({
        'Frontier_Size': ['mean', 'std', 'max'],
        'Iteration': 'count'
    }).round(2)
    
    # Performance indicators
    total_iterations = len(df)
    push_iterations = len(df[df['Mode'] == 'Push'])
    pull_iterations = len(df[df['Mode'] == 'Pull'])
    
    print("=== BFS Frontier Trajectory Analysis ===")
    print(f"Total iterations: {total_iterations}")
    print(f"Push mode iterations: {push_iterations} ({push_iterations/total_iterations*100:.1f}%)")
    print(f"Pull mode iterations: {pull_iterations} ({pull_iterations/total_iterations*100:.1f}%)")
    print(f"Mode switch count: {len(df[df['Mode'] != df['Mode'].shift()]) - 1}")
    print("\nMode statistics:")
    print(mode_stats)
    
    return mode_stats

def create_animated_frontier_plot(df):
    """
    Create an animated plot of frontier evolution (optional).
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
    
    # Animation generation is commented out to prevent issues in static environments
    # anim = FuncAnimation(fig, update, frames=len(df), interval=500, repeat=False)
    plt.close()
    
    return "Animation functionality prepared (uncomment related code to run in an interactive environment)"

# Main execution function
def main():
    """Main execution function"""
    print("Generating BFS Frontier trajectory data...")
    
    # Generate simulated data
    df = simulate_bfs_frontier_trajectory()
    
    # Plot trajectory
    print("Plotting trajectory...")
    fig = plot_frontier_trajectory(df)
    
    # Statistical analysis
    stats = create_comparison_analysis(df)
    
    # Save charts
    plt.savefig('bfs_frontier_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig('bfs_frontier_trajectory.pdf', bbox_inches='tight')
    
    # Display chart
    plt.show()
    
    # Save data
    df.to_csv('bfs_frontier_data.csv', index=False)
    
    print("\nChart saved as 'bfs_frontier_trajectory.png'")
    print("Data saved as 'bfs_frontier_data.csv'")
    
    return df, fig, stats

# Optional: create comparison under different parameters
def create_parameter_comparison():
    """Create comparison plots under different parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    density_thresholds = [0.01, 0.05, 0.1, 0.2]
    
    for i, threshold in enumerate(density_thresholds):
        np.random.seed(42)  # Keep other parameters consistent
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
    # Run main program
    df, fig, stats = main()
    
    # Optional: run parameter comparison
    # create_parameter_comparison()
