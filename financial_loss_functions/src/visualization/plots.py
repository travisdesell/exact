import os
import numpy as np
import matplotlib.pyplot as plt

def train_val_losses_plot(
    train_losses: list[float],
    val_losses: list[float],
    title: str,
    output_path: str,
    plot: bool = False,
    sharey: bool = False,          # set True to use same y-axis for easier comparison
    figsize: tuple = (12, 4)
):
    """Plot training and validation loss curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=sharey)

    # Left: train loss
    ax1.plot(train_losses, linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train Loss')
    ax1.grid(True)

    # Right: validation loss
    ax2.plot(val_losses, linestyle='-')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Validation Loss')
    ax2.grid(True)

    # Overall title centered above subplots
    fig.suptitle(title)

    # Tight layout so title and labels don't overlap
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and optionally show
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()

    plt.close('all')
    plt.clf()
    plt.cla()

def plot_windowed_comparison(
    all_daily_returns: dict,
    output_path: str,
    plot: bool=False
):
    
    cmap = plt.get_cmap('tab20') # or 'tab20', 'gist_rainbow'
    
    n_windows = next(iter(all_daily_returns.values())).shape[0]
    n_cols = min(3, n_windows)
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plotting loop for each window
    for window_idx in range(n_windows):
        ax = axes[window_idx]
        for i, (pf_type, daily_returns) in enumerate(all_daily_returns.items()):
            ax.plot(
                daily_returns[window_idx],
                label=pf_type,
                color=cmap(i % 20), # Using a 20-color map
                alpha=0.7,
                linewidth=1
            )
        ax.set_title(f'Window {window_idx + 1}')
        ax.grid(True, alpha=0.3)

    # 1. CLEAN UP MAIN PLOT
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # 2. CREATE SEPARATE LEGEND FILE
    # Extract handles and labels from the LAST used axis
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a small figure just for the legend
    # Adjust figsize based on how many portfolios you have
    fig_leg = plt.figure(figsize=(3, len(labels) * 0.2)) 
    legend = fig_leg.legend(handles, labels, loc='center', frameon=False, ncol=1)
    
    # Remove all axis info so it's just the legend
    plt.axis('off')
    
    # Generate legend path (e.g., 'path/to/plot_legend.png')
    base, ext = os.path.splitext(output_path)
    legend_path = f'{base}_legend{ext}'
    
    # Save with bbox_inches='tight' to crop the white space
    fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')

    if plot:
        plt.show()
    
    plt.close('all')