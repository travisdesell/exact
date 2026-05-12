import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go

# -------------------- Train - Val Loss Curves Plots -------------------- #
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

# -------------------- Windowed Returns Plots -------------------- #
def plot_windowed_comparison(
    all_daily_returns: dict,
    window_dates: list[pd.DatetimeIndex],  # Add this parameter (list of DatetimeIndex)
    windows_per_row: int,
    output_path: str | None = None,
    plot: bool = False
):
    cmap = plt.get_cmap('tab20')
    
    # Get number of windows from the first portfolio type
    # n_windows = next(iter(all_daily_returns.values())).shape[0]
    n_windows = len(next(iter(all_daily_returns.values())))
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

def plot_wind_comparison_str_dates(
    all_daily_returns: dict,
    window_dates: list[str],          # list of strings: 'YYYY-MM-DD_YYYY-MM-DD'
    windows_per_row: int,
    output_path: str | None = None,
    plot: bool = False
):
    cmap = plt.get_cmap('Set1')
    
    n_windows = len(next(iter(all_daily_returns.values())))
    n_cols = min(windows_per_row, n_windows)
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for window_idx in range(n_windows):
        ax = axes[window_idx]
        
        # Parse the date range string
        start_str, end_str = window_dates[window_idx].split('_')
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        
        # Generate business day date range (assumes trading days)
        date_range = pd.date_range(start, end, freq='B')
        
        for i, (pf_type, daily_returns) in enumerate(all_daily_returns.items()):
            y_data = daily_returns[window_idx]
            # In case date_range has more days than y_data (e.g., holidays), slice to same length
            x_vals = date_range[:len(y_data)]
            ax.plot(
                x_vals, y_data,
                label=pf_type,
                color=cmap(i % 20),
                alpha=0.7,
                linewidth=1
            )
        
        # Title using the original strings (already formatted)
        ax.set_title(f'W{window_idx+1}: {start_str} to {end_str}', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # Legend extraction (same as before)
    handles, labels = ax.get_legend_handles_labels()
    fig_leg = plt.figure(figsize=(3, len(labels)*0.2))
    fig_leg.legend(handles, labels, loc='center', frameon=False, ncol=1)
    plt.axis('off')
    if output_path:
        base, ext = os.path.splitext(output_path)
        legend_path = f'{base}_legend{ext}'
        fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')

    if plot:
        plt.show()
    
    plt.close('all')

# ------------------- Combined returns plot -------------------- #
def plot_concatenated_returns(
    all_daily_returns: dict,          # {model_name: np.array of concatenated returns}
    window_dates: list[str],          # e.g. ['2022-01-03_2022-03-29', ...]
    output_path: str | None = None,
    plot: bool = False,
    figsize: tuple = (12, 6),
    title: str = "Daily Net Returns (2022–2023 Test Period)"
):
    # Generate full date index
    full_dates = []
    for win_str in window_dates:
        start_str, end_str = win_str.split('_')
        dates = pd.date_range(start_str, end_str, freq='B')
        full_dates.extend(dates)
    full_dates = pd.DatetimeIndex(full_dates)
    
    # Align lengths
    first_returns = next(iter(all_daily_returns.values()))
    if len(first_returns) != len(full_dates):
        min_len = min(len(first_returns), len(full_dates))
        full_dates = full_dates[:min_len]
        all_daily_returns = {k: v[:min_len] for k, v in all_daily_returns.items()}
    
    # Plot
    cmap = plt.get_cmap('Set1')
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (model_name, returns) in enumerate(all_daily_returns.items()):
        ax.plot(full_dates, returns,
                label=model_name,
                color=cmap(i % 10),
                alpha=0.8,
                linewidth=1.2)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return (net of transaction costs)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    
    # Legend placed inside the plot (best location)
    ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize='small')
    
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    if plot:
        plt.show()
    plt.close(fig)

# -------------------- Model vs Metric Boxplot Plots -------------------- #
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

# -------------------- Correlation Plots -------------------- #
def plot_corr(corr: pd.DataFrame, title: str):
    # Plot heatmap
    plt.figure()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# -------------------- Pareto Frontier Plots -------------------- #
def plot_3d_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    dominated_col: str = 'dominated',
    title: str = None,
    x_title: str = None,
    y_title: str = None,
    z_title: str = None,
    marker_size_dominated: int = 5,
    marker_size_frontier: int = 10,
    show_line: bool = True,
    line_color: str = 'green',
    line_width: int = 2,
    line_dash: str = 'dash'
) -> go.Figure:
    """
    Create an interactive 3D Pareto frontier plot.

    Args:
        df: DataFrame containing models as index and metric columns.
        x_col, y_col, z_col: Names of the three metrics to plot.
        dominated_col: Column name that indicates dominated models (boolean).
        title: Plot title.
        x_title, y_title, z_title: Axis labels (defaults to column names).
        marker_size_dominated, marker_size_frontier: Marker sizes.
        show_line: Whether to connect frontier points with a line.
        line_color, line_width, line_dash: Line styling.

    Returns:
        plotly.graph_objects.Figure
    """
    # Create a copy to avoid modifying original
    plot_df = df.copy()
    # Ensure dominated_col exists
    if dominated_col not in plot_df.columns:
        raise ValueError(f"Column '{dominated_col}' not found in DataFrame. Run Pareto dominance first.")
    
    # Extract frontier and dominated
    frontier = plot_df[~plot_df[dominated_col]]
    dominated = plot_df[plot_df[dominated_col]]
    
    fig = go.Figure()
    
    # 1. Dominated models
    fig.add_trace(go.Scatter3d(
        x=dominated[x_col],
        y=dominated[y_col],
        z=dominated[z_col],
        mode='markers',
        marker=dict(size=marker_size_dominated, color='gray', opacity=0.6),
        text=dominated.index,
        hovertemplate='<b>%{text}</b><br>' +
                      f'{x_col}: %{{x:.3f}}<br>' +
                      f'{y_col}: %{{y:.3f}}<br>' +
                      f'{z_col}: %{{z:.3f}}<extra></extra>',
        name='Dominated'
    ))
    
    # 2. Frontier models
    fig.add_trace(go.Scatter3d(
        x=frontier[x_col],
        y=frontier[y_col],
        z=frontier[z_col],
        mode='markers',
        marker=dict(size=marker_size_frontier, color='green', symbol='diamond'),
        text=frontier.index,
        hovertemplate='<b>%{text}</b><br>' +
                      f'{x_col}: %{{x:.3f}}<br>' +
                      f'{y_col}: %{{y:.3f}}<br>' +
                      f'{z_col}: %{{z:.3f}}<extra></extra>',
        name='Pareto Frontier'
    ))
    
    # 3. Optional line connecting frontier points (sorted by x_col)
    if show_line and not frontier.empty:
        frontier_sorted = frontier.sort_values(x_col, ascending=False)
        fig.add_trace(go.Scatter3d(
            x=frontier_sorted[x_col],
            y=frontier_sorted[y_col],
            z=frontier_sorted[z_col],
            mode='lines',
            line=dict(color=line_color, width=line_width, dash=line_dash),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Layout
    fig.update_layout(
        title=title or f'3D Pareto Frontier ({x_col}, {y_col}, {z_col})',
        scene=dict(
            xaxis_title=x_title or x_col,
            yaxis_title=y_title or y_col,
            zaxis_title=z_title or z_col
        ),
        legend=dict(x=0.8, y=0.9)
    )
    
    return fig

def plot_2d_pareto(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    dominated_col: str = 'dominated',
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (8, 6),
    color_dominated: str = 'gray',
    color_frontier: str = 'green',
    alpha_dominated: float = 0.6,
    size_dominated: int = 80,
    size_frontier: int = 120,
    edgecolor_frontier: str = 'black',
    show_line: bool = True,
    line_style: str = '--',
    line_width: int = 2,
    annotate: bool = True,
    annotation_fontsize: int = 9,
    annotation_alpha: float = 0.8,
    grid: bool = True,
    grid_alpha: float = 0.3
) -> plt.Figure:
    """
    Plot a 2D Pareto frontier with optional frontier line and annotations.

    Args:
        df: DataFrame containing models as index and metric columns.
        x_col, y_col: Column names for the two metrics.
        dominated_col: Column name indicating dominated models (boolean).
        title, xlabel, ylabel: Plot labels.
        figsize: Figure size.
        marker_dominated, marker_frontier: Marker styles.
        color_dominated, color_frontier: Colors.
        alpha_dominated: Opacity for dominated points.
        size_dominated, size_frontier: Marker sizes.
        edgecolor_frontier: Edge color for frontier markers.
        show_line: Whether to draw a line connecting frontier points.
        line_style, line_width: Line style.
        annotate: Whether to add text labels for frontier points.
        annotation_fontsize, annotation_alpha: Text label styling.
        grid, grid_alpha: Grid settings.

    Returns:
        matplotlib.figure.Figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract dominated and frontier
    dominated = df[df[dominated_col]]
    frontier = df[~df[dominated_col]]
    
    # Plot dominated
    ax.scatter(dominated[x_col], dominated[y_col],
               c=color_dominated, marker='o', s=size_dominated,
               alpha=alpha_dominated, label='Dominated')
    
    # Plot frontier
    ax.scatter(frontier[x_col], frontier[y_col],
               c=color_frontier, marker='X', s=size_frontier,
               edgecolors=edgecolor_frontier, label='Pareto Frontier')
    
    # Optional frontier line (sorted by x_col)
    if show_line and not frontier.empty:
        frontier_sorted = frontier.sort_values(x_col, ascending=False)
        ax.plot(frontier_sorted[x_col], frontier_sorted[y_col],
                linestyle=line_style, color=color_frontier, linewidth=line_width,
                label='Frontier Line')
    
    # Annotations for frontier points
    if annotate:
        for _, row in frontier.iterrows():
            ax.annotate(row.name, (row[x_col], row[y_col]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=annotation_fontsize, alpha=annotation_alpha)
    
    # Labels and title
    ax.set_xlabel(xlabel if xlabel else x_col, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y_col, fontsize=12)
    ax.set_title(title if title else f'Pareto Frontier - {x_col} vs {y_col}', fontsize=14)
    ax.legend()
    if grid:
        ax.grid(True, alpha=grid_alpha)
    
    fig.tight_layout()
    return fig

# -------------------- Normality Plots -------------------- #
def plot_qq_by_model(
        all_seed_perf: pd.DataFrame,
        nn_model_losses,
        metrics=None,
        figsize=(5,4),
        n_cols=3
    ):
    """
    Draw Q-Q plots for each metric and each model-loss combination.
    
    Args:
        all_seed_perf : pd.DataFrame
            DataFrame with columns 'model_loss', seed index (or any identifier),
            and metric columns (e.g., 'sharpe', 'calmar', 'cvar', ...).
        nn_model_losses : list
            List of model-loss strings present in all_seed_perf['model_loss'].
        metrics : list, optional
            List of metric column names to plot. If None, all numeric columns except 'model_loss' are used.
        figsize : tuple, default (5,4)
            Size of each individual subplot.
        n_cols : int, default 3
            Number of columns in the faceted grid.
    """
    if metrics is None:
        # Use all numeric columns except the identifier column
        metrics = all_seed_perf.select_dtypes(include=[np.number]).columns.tolist()
        # Remove 'seed' if present (assuming no other non‑metric numeric columns)
        if 'seed' in metrics:
            metrics.remove('seed')
    
    n_models = len(nn_model_losses)
    n_metrics = len(metrics)
    
    for model in nn_model_losses:
        model_data = all_seed_perf[all_seed_perf.index == model]
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*figsize[0], n_rows*figsize[1]))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = model_data[metric].dropna().values
            if len(values) > 0:
                stats.probplot(values, dist="norm", plot=ax)
                ax.set_title(f"{model}\n{metric}")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{model}\n{metric}")
        # Hide any unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        fig.tight_layout()
        plt.show()

def plot_hist_by_model(
        all_seed_perf: pd.DataFrame,
        nn_model_losses,
        metrics=None,
        figsize=(5,4),
        n_cols=3,
        bins=20
    ):
    """
    Draw histograms for each metric and each model-loss combination.
    
    Args:
        all_seed_perf : pd.DataFrame
            DataFrame with columns 'model_loss', seed index (or any identifier),
            and metric columns (e.g., 'sharpe', 'calmar', 'cvar', ...).
        nn_model_losses : list
            List of model-loss strings present in all_seed_perf['model_loss'].
        metrics : list, optional
            List of metric column names to plot. If None, all numeric columns except 'model_loss' are used.
        figsize : tuple, default (5,4)
            Size of each individual subplot.
        n_cols : int, default 3
            Number of columns in the faceted grid.
        bins : int or sequence, default 20
            Number of histogram bins.
    """
    if metrics is None:
        metrics = all_seed_perf.select_dtypes(include=[np.number]).columns.tolist()
        if 'seed' in metrics:
            metrics.remove('seed')
    
    n_models = len(nn_model_losses)
    n_metrics = len(metrics)
    
    for model in nn_model_losses:
        model_data = all_seed_perf[all_seed_perf.index == model]
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*figsize[0], n_rows*figsize[1]))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = model_data[metric].dropna().values
            if len(values) > 0:
                ax.hist(values, bins=bins, alpha=0.7, edgecolor='black', density=True)
                ax.axvline(np.mean(values), color='red', linestyle='--', label='mean')
                ax.axvline(np.median(values), color='blue', linestyle='-.', label='median')
                ax.set_title(f"{model}\n{metric}")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_title(f"{model}\n{metric}")
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        fig.tight_layout()
        plt.show()

def plot_box_models_and_benchs(
    all_seed_perf: pd.DataFrame,
    bench_perfs: pd.DataFrame,
    metrics: list[str],
    figsize: tuple = (10, 6),
    palette: str = 'Set2',
    benchmark_marker: str = 'D',
    benchmark_markersize: int = 8,
    output_path: str | None = None
):
    """
    Create a vertically stacked figure with one subplot per metric.
    Each subplot shows boxplots for neural models (30 seeds) and markers
    for deterministic benchmarks as separate x-axis categories.

    Args:
        all_seed_perf : pd.DataFrame
            Index: model names (repeated 30 times per model). Columns: metric values.
        bench_perfs : pd.DataFrame
            Index: benchmark names (e.g., 'S&P500', 'Equal_Weight', 'NCO').
            Columns: same metrics.
        metrics : list[str]
            List of metric column names to plot (e.g., ['sharpe', 'calmar', 'neg_cvar']).
        figsize : tuple, default (10,6)
            Size of each subplot (width, height). Total figure height scales with number of metrics.
        palette : str, default 'Set2'
            Seaborn colour palette for neural model boxplots.
        benchmark_marker : str, default 'D'
            Marker style for benchmark points.
        benchmark_markersize : int, default 8
            Size of benchmark markers.
        output_path (str | None) : Output path to save the image. Default = None.
    """
    # Prepare neural data: convert index to column
    df_neural = all_seed_perf.reset_index().rename(columns={'index': 'model'})
    # Prepare benchmark data
    bench_long = bench_perfs.reset_index().rename(columns={'index': 'model'})
    # Get ordered lists
    neural_models = df_neural['model'].unique()
    benchmark_models = bench_long['model'].unique()
    all_categories = list(neural_models) + list(benchmark_models)

    n_metrics = len(metrics)
    # Create figure with n_metrics subplots (one per row)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(figsize[0], figsize[1] * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Boxplot for neural models (only)
        sns.boxplot(data=df_neural, x='model', y=metric, ax=ax,
                    palette=palette, order=neural_models, hue='model', legend=False,
                    showfliers=False, linewidth=1)
        
        # Add benchmark points as separate x‑positions
        # Get current x positions (0,1,... for neural models)
        current_ticks = np.arange(len(neural_models))
        # Extend ticks to include benchmarks
        new_ticks = np.append(current_ticks, current_ticks[-1] + 1 + np.arange(len(benchmark_models)))
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(all_categories, rotation=45, ha='right')
        
        # Plot benchmark points
        for i, bench_name in enumerate(benchmark_models):
            if metric not in bench_long.columns:
                continue
            bench_val = bench_long[bench_long['model'] == bench_name][metric].values[0]
            x_pos = len(neural_models) + i
            ax.scatter(x_pos, bench_val, marker=benchmark_marker,
                       s=benchmark_markersize**2, color='red', edgecolor='black',
                       zorder=10, label=bench_name if i == 0 else "")
        
        # Add legend for benchmarks (only once)
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate benchmark labels
        # unique = dict(zip(labels, handles))
        # if unique:
        #     ax.legend(unique.values(), unique.keys(), loc='upper left', fontsize=8)
        
        ax.set_title(f'{metric.upper()} Distribution')
        ax.set_xlabel('Model / Benchmark')
        if metric == 'cvar':
            ax.set_ylabel('CVaR (%)')
        else:
            ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        # Adjust y‑axis limits if needed (optional)
        # ax.autoscale(enable=True, axis='y')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_2d_pareto_diff_colrs(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    dominated_col: str = 'dominated',
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    figsize: tuple = (8, 6),
    color_dominated: str = 'gray',
    alpha_dominated: float = 0.6,
    size_dominated: int = 80,
    size_frontier: int = 120,
    edgecolor_frontier: str = 'black',
    edgewidth_frontier: float = 1.5,
    show_line: bool = True,
    line_style: str = '--',
    line_width: int = 2,
    line_color: str = 'green',
    annotate: bool = False,
    annotation_fontsize: int = 9,
    annotation_alpha: float = 0.8,
    grid: bool = True,
    grid_alpha: float = 0.3,
    colormap: str = 'Set1',   # better contrast default
    output_path: str | None = None
) -> plt.Figure:
    """
    Plot a 2D Pareto frontier with distinct colors for each frontier model.
    Frontier line is drawn behind points. Model names appear in the legend.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with models as index, containing x_col, y_col, and dominated_col.
    x_col, y_col : str
        Column names for the two metrics.
    dominated_col : str
        Boolean column indicating dominated models.
    title, xlabel, ylabel : str, optional
        Plot labels.
    figsize : tuple, default (8,6)
        Figure size.
    color_dominated : str, default 'gray'
        Color for dominated points.
    alpha_dominated : float, default 0.6
        Transparency for dominated points.
    size_dominated, size_frontier : int
        Marker sizes.
    edgecolor_frontier : str, default 'black'
        Edge color for frontier markers.
    edgewidth_frontier : float, default 1.5
        Edge linewidth for frontier markers.
    show_line : bool, default True
        Whether to draw a line connecting frontier points.
    line_style, line_width, line_color : str, int, str
        Line style, width, and color.
    annotate : bool, default False
        If True, add text labels next to frontier points.
    annotation_fontsize, annotation_alpha : int, float
        Font size and transparency for annotations.
    grid, grid_alpha : bool, float
        Grid settings.
    colormap : str or list
        Name of a matplotlib colormap (e.g., 'Set1', 'tab10', 'viridis') or a list of colors.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dominated = df[df[dominated_col]]
    frontier = df[~df[dominated_col]]

    # Step 1: Draw frontier line (if requested) – behind points
    if show_line and not frontier.empty:
        frontier_sorted = frontier.sort_values(x_col, ascending=False)
        ax.plot(frontier_sorted[x_col], frontier_sorted[y_col],
                linestyle=line_style, color=line_color, linewidth=line_width,
                label='Frontier Line', zorder=1)

    # Step 2: Dominated points
    ax.scatter(dominated[x_col], dominated[y_col],
               c=color_dominated, marker='o', s=size_dominated,
               alpha=alpha_dominated, label='Dominated', zorder=2)

    # Step 3: Assign distinct colors to frontier models
    if isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i) for i in np.linspace(0, 1, len(frontier))]
    else:
        colors = colormap[:len(frontier)]

    # Step 4: Plot each frontier point (on top)
    for (idx, row), color in zip(frontier.iterrows(), colors):
        ax.scatter(row[x_col], row[y_col],
                   c=color, marker='X', s=size_frontier,
                   edgecolors=edgecolor_frontier, linewidth=edgewidth_frontier,
                   label=row.name, zorder=3)

    # Step 5: Optional annotations
    if annotate:
        for _, row in frontier.iterrows():
            ax.annotate(row.name, (row[x_col], row[y_col]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=annotation_fontsize, alpha=annotation_alpha)

    # Labels and title
    ax.set_xlabel(xlabel if xlabel else x_col, fontsize=12)
    ax.set_ylabel(ylabel if ylabel else y_col, fontsize=12)
    ax.set_title(title if title else f'Pareto Frontier - {x_col} vs {y_col}', fontsize=14)

    # Legend: remove duplicate entries if any
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='best', fontsize=10)

    if grid:
        ax.grid(True, alpha=grid_alpha)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig