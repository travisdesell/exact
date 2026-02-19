import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    window_dates: list,  # Add this parameter (list of DatetimeIndex)
    output_path: str,
    plot: bool = False
):
    cmap = plt.get_cmap('tab20')
    
    # Get number of windows from the first portfolio type
    n_windows = next(iter(all_daily_returns.values())).shape[0]
    n_cols = min(3, n_windows)
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for window_idx in range(n_windows):
        ax = axes[window_idx]
        
        # Get the specific dates for this window
        current_window_dates = window_dates[window_idx]
        
        for i, (pf_type, daily_returns) in enumerate(all_daily_returns.items()):
            # Ensure the data matches the date length
            y_data = daily_returns[window_idx]
            
            # Use the dates as the X-axis
            ax.plot(
                current_window_dates, 
                y_data,
                label=pf_type,
                color=cmap(i % 20),
                alpha=0.7,
                linewidth=1
            )
        
        # Format the Title with Start/End dates for better context
        start_str = current_window_dates[0].strftime('%Y-%m-%d')
        end_str = current_window_dates[-1].strftime('%Y-%m-%d')
        ax.set_title(f'W{window_idx + 1}: {start_str} to {end_str}', fontsize=10)
        
        # Add date formatting to make them readable
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) # Shows Month-Day
        ax.grid(True, alpha=0.3)

    # Clean up empty subplots
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    # MAGIC STEP: Automatically rotates and aligns the tick labels
    fig.autofmt_xdate()
    
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