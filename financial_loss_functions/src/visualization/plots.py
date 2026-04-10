import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def train_val_losses_plot(
    train_losses: list[float],
    val_losses: list[float],
    eval_losses: list[float], # The individual losses for Window 1, Window 2, etc.
    title: str,
    output_path: str,
    plot: bool = False,
    figsize: tuple = (14, 5)
):
    """
    Left: Training vs Validation curves over Epochs.
    Right: Bar chart of specific Out-of-Sample Window losses.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: Combined Loss Curves (Epoch-based) ---
    epochs = range(len(train_losses))
    ax1.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss (Avg)', color='#ff7f0e', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves (Loss vs Epoch)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Right: Column Chart (Window-based) ---
    num_windows = len(eval_losses)
    window_labels = [f'Window {i+1}' for i in range(num_windows)]
    
    bars = ax2.bar(window_labels, eval_losses)
    
    # Add value labels on top of the bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax2.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    ax2.set_ylabel('Loss')
    ax2.set_title(f'Evaluation Performance (N={num_windows} Windows)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close('all')
    plt.clf()
    plt.cla()

def wfv_losses_plot(
        wfv_train: list[list[float]],
        wfv_eval_losses: list[float],
        title: str,
        output_path: str,
        plot: bool = False,
        figsize: tuple = (14, 5)
    ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for i, step_trains in enumerate(wfv_train):
        epochs = range(len(step_trains))
        ax1.plot(epochs, step_trains, label=f'Step {i}', linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves (Loss vs Epoch)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    # --- Right: Column Chart (Window-based) ---
    num_windows = len(wfv_eval_losses)
    window_labels = [f'Step {i}' for i in range(num_windows)]
    
    bars = ax2.bar(window_labels, wfv_eval_losses)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    ax2.set_ylabel('Loss')
    ax2.set_title(f'Evaluation Performance (N={num_windows} Walk Steps)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close('all')
    plt.clf()
    plt.cla()

def plot_windowed_comparison(
    all_daily_returns: dict,
    window_dates: list,  # Add this parameter (list of DatetimeIndex)
    windows_per_row: int,
    output_path: str | None = None,
    plot: bool = False
):
    cmap = plt.get_cmap('tab20')
    
    # Get number of windows from the first portfolio type
    n_windows = next(iter(all_daily_returns.values())).shape[0]
    n_cols = min(windows_per_row, n_windows)
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

    if output_path:
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
    
    if output_path:
        # Generate legend path (e.g., 'path/to/plot_legend.png')
        base, ext = os.path.splitext(output_path)
        legend_path = f'{base}_legend{ext}'
        
        # Save with bbox_inches='tight' to crop the white space
        fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')

    if plot:
        plt.show()
    
    plt.close('all')

def plot_models_comparison(
        eval_metrics: pd.DataFrame,
        title: str,
        output_path: str,
        plot: bool = False
    ):
    
    eval_metrics.boxplot(figsize=(12, 6))
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout() # Ensures labels aren't cut off


    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if plot:
        plt.show()

    plt.close('all')