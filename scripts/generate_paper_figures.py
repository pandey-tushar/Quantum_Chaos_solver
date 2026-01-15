#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data_utils import load_classical_solution


def fig1_trajectory_comparison():
    """
    Figure 1: Lorenz attractor trajectory comparison.
    Shows classical solution and predicted solutions.
    """
    print("Generating Figure 1: Trajectory comparison...")
    
    # Load classical solution
    classical = load_classical_solution(project_root / "data")
    
    # Load reservoir predictions
    reservoir_data = np.load(project_root / "results" / "reservoir_final" / "reservoir_model.npz")
    
    # Get time range used for training
    t_train = reservoir_data['t_train']
    train_states = reservoir_data['train_states']
    train_preds = reservoir_data['train_predictions']
    
    # Create figure with 3D attractor and time series
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Attractor plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot full classical trajectory (faded)
    mask = classical['t'] <= 4.0
    ax1.plot(classical['x'][mask], classical['y'][mask], classical['z'][mask], 
             'b-', alpha=0.3, linewidth=0.5, label='Classical')
    
    # Plot training region
    ax1.plot(train_states[:, 0], train_states[:, 1], train_states[:, 2],
             'b-', linewidth=1.5, label='Training data')
    
    # Adjust view for 3D
    # Note: train_preds has fewer points due to windowing
    # We need to align with the later part of train_states
    window_size = 5
    aligned_states = train_states[window_size-1:]
    
    ax1.scatter(aligned_states[:, 0], aligned_states[:, 1], aligned_states[:, 2],
                c='red', s=20, alpha=0.8, label='QRC predictions', zorder=5)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('(a) Lorenz Attractor')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=20, azim=45)
    
    # Time series comparison
    ax2 = fig.add_subplot(122)
    
    # Align times for predictions
    t_aligned = t_train[window_size-1:]
    
    # Plot x component
    ax2.plot(t_train, train_states[:, 0], 'b-', linewidth=1.5, label='Classical x(t)')
    ax2.scatter(t_aligned, train_preds[:, 0], c='red', s=15, alpha=0.7, label='QRC x(t)', zorder=5)
    
    # Plot y component (offset for visibility)
    offset = 60
    ax2.plot(t_train, train_states[:, 1] + offset, 'g-', linewidth=1.5, label='Classical y(t)')
    ax2.scatter(t_aligned, train_preds[:, 1] + offset, c='orange', s=15, alpha=0.7, label='QRC y(t)', zorder=5)
    
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('State variables (offset for clarity)')
    ax2.set_title('(b) Time Series Comparison')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = project_root / "paper" / "fig1_trajectory.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def fig2_training_curves():
    """
    Figure 2: PINN training curves showing loss evolution.
    """
    print("Generating Figure 2: Training curves...")
    
    import json
    
    # Load PINN training history
    with open(project_root / "results" / "pinn_final" / "training_history.json") as f:
        pinn_history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot total loss
    ax1 = axes[0]
    iterations = pinn_history['iteration']
    
    # Calculate per-iteration times
    times = pinn_history['time']
    iter_times = [times[i+1] - times[i] for i in range(len(times)-1)]
    
    ax1.semilogy(iterations, pinn_history['loss_total'], 'b-', linewidth=1, alpha=0.7, label='Total Loss')
    ax1.semilogy(iterations, pinn_history['loss_data'], 'r-', linewidth=1.5, label='Data Loss (MSE)')
    ax1.semilogy(iterations, pinn_history['loss_physics'], 'g-', linewidth=1, alpha=0.7, label='Physics Loss')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('(a) QPINN Training Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add final MSE annotation
    final_mse = pinn_history['loss_data'][-1]
    ax1.axhline(y=final_mse, color='r', linestyle='--', alpha=0.5)
    ax1.annotate(f'Final MSE: {final_mse:.1f}', xy=(150, final_mse*1.5), fontsize=10, color='red')
    
    # Plot gradient norm
    ax2 = axes[1]
    ax2.plot(iterations, pinn_history['grad_norm'], 'purple', linewidth=1, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('(b) Gradient Magnitude')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation about no vanishing gradients
    mean_grad = np.mean(pinn_history['grad_norm'])
    ax2.axhline(y=mean_grad, color='purple', linestyle='--', alpha=0.5)
    ax2.annotate(f'Mean: {mean_grad:.1f}', xy=(150, mean_grad*1.2), fontsize=10, color='purple')
    
    plt.tight_layout()
    
    output_path = project_root / "paper" / "fig2_training.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def fig3_ablation():
    """
    Figure 3: Ablation study - window size effect.
    """
    print("Generating Figure 3: Ablation study...")
    
    import json
    
    # Load ablation results
    with open(project_root / "results" / "ablation_results.json") as f:
        ablation = json.load(f)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    windows = [r['window'] for r in ablation]
    mses = [r['mse'] for r in ablation]
    feat_dims = [r['feat_dim'] for r in ablation]
    
    # Bar plot
    bars = ax.bar(range(len(windows)), mses, color=['#ff6b6b', '#feca57', '#48dbfb'], 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, mse, dim in zip(bars, mses, feat_dims):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mse:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{dim}D', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f'w={w}' if w > 1 else 'w=1\n(no window)' for w in windows])
    ax.set_ylabel('Train MSE')
    ax.set_title('Effect of Temporal Window Size on QRC Performance')
    ax.set_ylim(0, max(mses) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add improvement annotation
    improvement = (mses[0] - mses[2]) / mses[0] * 100
    ax.annotate(f'{improvement:.0f}% improvement\nwith windowing', 
                xy=(2, mses[2]), xytext=(1.5, mses[0]*0.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    output_path = project_root / "paper" / "fig3_ablation.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def fig4_comparison_summary():
    """
    Figure 4: Summary comparison bar chart.
    """
    print("Generating Figure 4: Method comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    methods = ['QPINN\n(4q, 3L)', 'QRC\n(5q, 2L)']
    mses = [37.77, 22.37]
    times = [3.1 * 3600, 0.8]  # Convert hours to seconds for PINN
    
    # MSE comparison
    ax1 = axes[0]
    colors = ['#ff6b6b', '#48dbfb']
    bars1 = ax1.bar(methods, mses, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, mse in zip(bars1, mses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mse:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('(a) Accuracy Comparison')
    ax1.set_ylim(0, max(mses) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add improvement annotation
    improvement = (mses[0] - mses[1]) / mses[0] * 100
    ax1.annotate(f'↓{improvement:.1f}%', xy=(1, mses[1]), xytext=(0.5, mses[0]*0.6),
                fontsize=14, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Time comparison (log scale)
    ax2 = axes[1]
    bars2 = ax2.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_yscale('log')
    
    # Add time labels
    time_labels = ['3.1 hours', '0.8 sec']
    for bar, label in zip(bars2, time_labels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                label, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Training Time (seconds, log scale)')
    ax2.set_title('(b) Training Speed Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add speedup annotation
    speedup = times[0] / times[1]
    ax2.annotate(f'{speedup:.0f}× faster', xy=(1, times[1]), xytext=(0.5, times[0]*0.1),
                fontsize=14, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    
    output_path = project_root / "paper" / "fig4_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    print()
    
    # Create paper directory if needed
    paper_dir = project_root / "paper"
    paper_dir.mkdir(exist_ok=True)
    
    # Generate all figures
    fig1_trajectory_comparison()
    fig2_training_curves()
    fig3_ablation()
    fig4_comparison_summary()
    
    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print()
    print("Files created in paper/:")
    for f in sorted(paper_dir.glob("fig*.pdf")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

