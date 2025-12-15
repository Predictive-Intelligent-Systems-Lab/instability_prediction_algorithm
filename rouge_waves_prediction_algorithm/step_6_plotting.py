import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set consistent figure size for all plots
FIGURE_WIDTH = 15
FIGURE_HEIGHT = 6
FIGURE1_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)  # Signal plot
FIGURE2_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)  # Histogram plot
FIGURE3_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)  # Confidence histogram
FIGURE4_SIZE = (FIGURE_WIDTH, FIGURE_HEIGHT)  # Detection performance viz

def plot_original_signal(data_file='RWs_H_g_2p2_tadv_1min_buoy_132.npz', 
                         time_range=(0, 1500), 
                         y_range=None):
    """
    Load and plot the original signal from the NPZ file before any transformations.
    
    Parameters:
    -----------
    data_file : str
        Path to the NPZ file containing the wave data
    time_range : tuple, optional
        Time range to plot (min_time, max_time) in minutes
        Default is (0, 1500)
    y_range : tuple, optional
        Y-axis range (min_y, max_y)
        Default is None (auto-determined)
    """
    # Create graphical_results directory for saving plots
    script_dir = Path(__file__).parent
    save_dir = script_dir / "graphical_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {data_file}")
    print(f"Saving graphs to: {save_dir}")
    
    # Set global font properties to Arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 30  # Base font size for tick labels
    
    # Prevent axis label overlap by increasing padding
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['xtick.major.pad'] = 7
    plt.rcParams['ytick.major.pad'] = 7
    
    # Load the NPZ file
    try:
        data = np.load(data_file)
        print("Keys in the .npz file:", data.files)
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found!")
        return
    
    # Extract original signal (same as in first script)
    original_signal = data['wave_data_train'][0].squeeze()
    
    # Create time array (1 minute per sample)
    dt = 1.0  # minutes
    t_physical = np.arange(0, len(original_signal) * dt, dt)
    
    print(f"Signal length: {len(original_signal)}")
    print(f"Time range: {t_physical[0]:.2f} to {t_physical[-1]:.2f} minutes")
    
    # PLOT: Original Signal
    plt.figure(figsize=FIGURE1_SIZE)
    
    # Create a figure with more precise control over margins
    fig = plt.gcf()
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)
    
    plt.plot(t_physical, original_signal, color='blue', linewidth=1.2)
    
    # Add x and y labels
    plt.xlabel('Time (minutes)', fontsize=28)
    plt.ylabel('Sea surface elevation', fontsize=28)
    
    plt.grid(alpha=0.3)
    
    # Set time range to specified range
    plt.xlim(time_range[0], time_range[1])
    
    # Get the y-range for the visible portion of the signal
    visible_indices = np.where((t_physical >= time_range[0]) & (t_physical <= time_range[1]))[0]
    if len(visible_indices) > 0:
        visible_signal = original_signal[visible_indices]
        # Use provided y_range if available, otherwise calculate automatically
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        else:
            y_min = visible_signal.min() * 0.95
            y_max = visible_signal.max() * 1.05
            plt.ylim(y_min, y_max)
    
    # Use consistent padding
    plt.tight_layout(pad=1.5)
    
    # Save as JPG
    save_path = save_dir / f"original_signal_{time_range[0]}_to_{time_range[1]}.jpg"
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format='jpg')
    print(f"Saved: {save_path}")
    plt.show()
    
    print("\nPlotting complete!")

def plot_signal_data(data_dir, time_range=(0, 1500), y_range=None):
    """
    Read the saved NPZ files and create visualizations for the signal data.
    All graphs are now saved as .jpg instead of .png
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory where the data files are stored
    time_range : tuple, optional
        Time range to plot for the first graph (min_time, max_time)
        Default is set to (0, 1500) time units
    y_range : tuple, optional
        Y-axis range for the first graph (min_y, max_y)
        Default is None (auto-determined based on visible data)
    """
    data_dir = Path(data_dir)
    
    # Create graphical_results directory for saving plots
    script_dir = Path(__file__).parent
    save_dir = script_dir / "graphical_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data from: {data_dir}")
    print(f"Saving graphs to: {save_dir}")
    
    # Set global font properties to Arial
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 30  # Base font size for tick labels
    
    # Prevent axis label overlap by increasing padding
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['xtick.major.pad'] = 7
    plt.rcParams['ytick.major.pad'] = 7
    
    # Define a consistent color for bar plots
    bar_color = 'skyblue'
    
    # Load all data files
    signal_data = np.load(data_dir / "signal_data.npz")
    peak_data = np.load(data_dir / "peak_data.npz")
    precursor_data = np.load(data_dir / "precursor_data.npz")
    warning_time_data = np.load(data_dir / "warning_time_data.npz")
    prediction_data = np.load(data_dir / "prediction_data.npz")
    
    # Load prediction data if it exists
    try:
        type2_data = np.load(data_dir / "type2_prediction_data.npz")
        has_type2_data = True
    except FileNotFoundError:
        has_type2_data = False
    
    # Extract data
    t_physical = signal_data['time']
    full_signal = signal_data['signal']
    threshold = signal_data['threshold']
    
    # Extract time unit information if available (use defaults if not)
    time_unit = signal_data['time_unit'] if 'time_unit' in signal_data else 'time units'
    time_unit_abbr = signal_data['time_unit_abbr'] if 'time_unit_abbr' in signal_data else 't.u.'
    
    # Get axis labels if available
    x_label_signal = signal_data['x_label'] if 'x_label' in signal_data else 'Time'
    y_label_signal = signal_data['y_label'] if 'y_label' in signal_data else 'Signal Amplitude'
    
    # Get warning time x-label if available
    x_label_warning = warning_time_data['x_label'] if 'x_label' in warning_time_data else 'Warning Time'
    
    extreme_indices = peak_data['extreme_indices'] if 'extreme_indices' in peak_data else []
    extreme_values = peak_data['extreme_values'] if 'extreme_values' in peak_data else []
    
    # Get non-extreme peaks if available
    non_extreme_indices = peak_data['non_extreme_indices'] if 'non_extreme_indices' in peak_data else []
    
    precursor_times = precursor_data['precursor_times'] if 'precursor_times' in precursor_data else []
    peak_indices = precursor_data['peak_indices'] if 'peak_indices' in precursor_data else []
    peak_times = precursor_data['peak_times'] if 'peak_times' in precursor_data else []
    precursor_values = precursor_data['precursor_values'] if 'precursor_values' in precursor_data else []
    
    delta_t_values = warning_time_data['delta_t_values'] if 'delta_t_values' in warning_time_data else []
    
    # Extract prediction data
    type1_times = prediction_data['type1_times'] if 'type1_times' in prediction_data else []
    type1_confidences = prediction_data['type1_confidences'] if 'type1_confidences' in prediction_data else []
    
    # PLOT 1: Signal with precursors (zoomed to specified range)
    plt.figure(figsize=FIGURE1_SIZE)
    
    # Create a figure with more precise control over margins
    fig1 = plt.gcf()
    fig1.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)
    
    plt.plot(t_physical, full_signal, color='blue', linewidth=1.2)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.0)
    
    # Mark extreme peaks
    if len(extreme_indices) > 0:
        plt.scatter(t_physical[extreme_indices.astype(int)], 
                   full_signal[extreme_indices.astype(int)], 
                   color='red', s=80)
    
    # Mark precursors with yellow dots
    if len(precursor_times) > 0:
        plt.scatter(precursor_times, precursor_values, 
                   color='yellow', s=120, edgecolor='black')
    
    # Add x and y labels
    plt.xlabel(x_label_signal, fontsize=28)
    plt.ylabel(y_label_signal, fontsize=28)
    
    plt.grid(alpha=0.3)
    
    # Set time range to specified range
    plt.xlim(time_range[0], time_range[1])
    
    # Get the y-range for the visible portion of the signal
    visible_indices = np.where((t_physical >= time_range[0]) & (t_physical <= time_range[1]))[0]
    if len(visible_indices) > 0:
        visible_signal = full_signal[visible_indices]
        # Use provided y_range if available, otherwise calculate automatically
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        else:
            y_min = min(visible_signal.min(), threshold) * 0.95
            y_max = max(visible_signal.max(), threshold) * 1.05
            plt.ylim(y_min, y_max)
    
    # Use consistent padding for all figures with bbox_inches='tight'
    plt.tight_layout(pad=1.5)
    
    # Save Figure 1 as JPG
    plt.savefig(save_dir / f"signal_plot_{time_range[0]}_to_{time_range[1]}.jpg", 
                dpi=600, bbox_inches='tight', format='jpg')
    plt.show()
    
    # PLOT 2: PDF of time differences (delta_t)
    plt.figure(figsize=FIGURE2_SIZE)
    
    # Create a figure with more precise control over margins
    fig2 = plt.gcf()
    fig2.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)
    
    # Create histogram with a consistent bar width
    bins = min(20, len(delta_t_values))  # Adjust bin number based on data size
    if len(delta_t_values) > 0:  # Only create histogram if there are values
        # Calculate histogram data first to determine bin width
        counts, bin_edges = np.histogram(delta_t_values, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        # Then plot with the calculated bin width
        plt.bar(bin_edges[:-1], counts, width=bin_width * 0.8, align='edge', 
                alpha=0.7, color=bar_color, edgecolor='black')
        
        # Add vertical line for mean - red color
        mean_dt = np.mean(delta_t_values)
        plt.axvline(x=mean_dt, color='red', linestyle='--', linewidth=2)
    else:
        plt.text(0.5, 0.5, "No data available for histogram", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=25)
    
    # Add x and y labels
    plt.xlabel(x_label_warning, fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    
    plt.grid(alpha=0.3)
    
    # Use consistent padding for all figures with bbox_inches='tight'
    plt.tight_layout(pad=1.5)
    
    # Save Figure 2 as JPG
    plt.savefig(save_dir / "warning_time_histogram.jpg", dpi=600, bbox_inches='tight', format='jpg')
    plt.show()
    
    # PLOT 3: Confidence histogram for Type I (2nd quadrant) predictions
    plt.figure(figsize=FIGURE3_SIZE)
    
    # Create a figure with more precise control over margins
    fig3 = plt.gcf()
    fig3.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)
    
    # Create histogram if we have data
    if len(type1_confidences) > 0:
        # Create histogram data (20 bins from 0 to 1)
        bins = np.linspace(0, 1, 21)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_width = bin_centers[1] - bin_centers[0]
        
        # Create histogram with the same color as Fig 2
        plt.bar(bin_centers, np.histogram(type1_confidences, bins=bins, density=True)[0], 
                width=bar_width * 0.8, alpha=0.7, color=bar_color, edgecolor='black')
        
        # Add vertical line for mean
        mean_conf = np.mean(type1_confidences)
        plt.axvline(x=mean_conf, color='red', linestyle='--', linewidth=2)
        
    else:
        plt.text(0.5, 0.5, "No Type I (2nd quadrant) predictions available", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=25)
    
    # Add x and y labels
    plt.xlabel('Confidence', fontsize=28)
    plt.ylabel('Probability Density', fontsize=28)
    
    plt.grid(alpha=0.3)
    
    # Set x-axis to 0-1 range for confidence
    plt.xlim(0, 1)
    
    # Use consistent padding for all figures with bbox_inches='tight'
    plt.tight_layout(pad=1.5)
    
    # Save Figure 3 as JPG
    plt.savefig(save_dir / "confidence_histogram.jpg", dpi=600, bbox_inches='tight', format='jpg')
    plt.show()
    
    # PLOT 4: Detection performance visualization
    if len(extreme_indices) > 0:
        fig, ax = plt.subplots(figsize=FIGURE4_SIZE)
        
        # Apply consistent margins
        fig.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.95)
        
        # Identify which extreme events were detected (had precursors) and which were missed
        detected_indices = set(peak_indices.astype(int)) if len(peak_indices) > 0 else set()
        all_extreme_indices = set(extreme_indices.astype(int))
        missed_indices = all_extreme_indices - detected_indices
        
        # Calculate percentages
        total_extreme_events = len(all_extreme_indices)
        detected_percentage = (len(detected_indices) / total_extreme_events * 100) if total_extreme_events > 0 else 0
        missed_percentage = (len(missed_indices) / total_extreme_events * 100) if total_extreme_events > 0 else 0
        
        # Create a visual representation with consistent bar width
        bar_width = 0.2
        
        detected_bar = plt.bar(0, len(detected_indices), width=bar_width, color='green', alpha=0.7, edgecolor='black')
        missed_bar = plt.bar(1, len(missed_indices), width=bar_width, color='red', alpha=0.7, edgecolor='black')
        
        # Add percentage labels on top of bars
        if total_extreme_events > 0:
            plt.text(0, len(detected_indices) + 0.5, f"{detected_percentage:.1f}%", 
                    ha='center', va='bottom', fontsize=24)
            plt.text(1, len(missed_indices) + 0.5, f"{missed_percentage:.1f}%", 
                    ha='center', va='bottom', fontsize=24)
        
        # Set x-axis labels
        plt.xticks([0, 1], ['Detected\n(with precursors)', 'Missed\n(no precursors)'], fontsize=24)
        
        # Set y-axis label on two lines
        plt.ylabel('Number of\nExtreme Events', fontsize=28)
        
        # Add grid with same alpha as other plots
        plt.grid(alpha=0.3)
        
        # Add the box around the plot by setting all spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        # Adjust y limit to allow space for percentage labels
        max_count = max(len(detected_indices), len(missed_indices))
        plt.ylim(0, max_count * 1.2)
        
        # Use consistent padding for all figures with bbox_inches='tight'
        plt.tight_layout(pad=1.5)
        
        # Save Figure 4 as JPG
        plt.savefig(save_dir / "detection_performance.jpg", dpi=600, bbox_inches='tight', format='jpg')
        plt.show()
    
    # Print summary information
    print("\nPlotting Summary:")
    print(f"Full time range: {t_physical[0]:.2f} to {t_physical[-1]:.2f} {time_unit_abbr}")
    print(f"Zoomed view: {time_range[0]:.2f} to {time_range[1]:.2f} {time_unit_abbr}")
    print(f"Extreme events detected: {len(extreme_indices)}")
    print(f"Precursors identified: {len(precursor_times)}")
    print(f"All graphs saved successfully to: {save_dir}")
    
    # Count how many extreme events and precursors are within the zoomed range
    extreme_in_range = sum(1 for idx in extreme_indices if time_range[0] <= t_physical[int(idx)] <= time_range[1])
    precursors_in_range = sum(1 for t in precursor_times if time_range[0] <= t <= time_range[1])
    print(f"Extreme events in zoomed view: {extreme_in_range}")
    print(f"Precursors in zoomed view: {precursors_in_range}")
    
    if len(delta_t_values) > 0:
        print(f"\nWarning time statistics:")
        print(f"Mean warning time: {np.mean(delta_t_values):.2f} {time_unit_abbr}")
        print(f"Median warning time: {np.median(delta_t_values):.2f} {time_unit_abbr}")
        print(f"Min/Max warning time: {np.min(delta_t_values):.2f}/{np.max(delta_t_values):.2f} {time_unit_abbr}")

# Example usage:
if __name__ == "__main__":
    # Use the same directory as the script
    script_dir = Path(__file__).parent
    
    # OPTION 1: Plot the original signal directly from the NPZ file
    print("=" * 60)
    print("PLOTTING ORIGINAL SIGNAL (before transformations)")
    print("=" * 60)
    plot_original_signal('RWs_H_g_2p2_tadv_1min_buoy_132.npz', 
                         time_range=(0, 1500), 
                         y_range=(-0.7, 1.1))
    
    # OPTION 2: Plot the processed signal data (after transformations)
    print("\n" + "=" * 60)
    print("PLOTTING PROCESSED SIGNAL (after transformations)")
    print("=" * 60)
    plot_signal_data(script_dir / "graphical_results", time_range=(0, 1500), y_range=(-0.1, 1.5))