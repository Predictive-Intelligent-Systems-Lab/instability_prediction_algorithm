import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from scipy.signal import find_peaks, argrelextrema
from tqdm import tqdm

# =========================
# Global Parameters - ALL PARAMETERS IN ONE PLACE
# =========================
PARAMS = {
    # Signal parameters
    'dt': 1.0,              # Time step value (1 minute per sample for experimental data)
    'window_size': 150,     # Window size for segmentation
    'overlap': 135,          # Overlap between segments
    't_span': None,         # Time range for analysis (set dynamically)
    'threshold_factor': 10.0, # Number of standard deviations above mean for extreme event classification
    # 'max_lookback': 50,    # Max lookback window for precursor identification
    'max_lookback': 25,    # Max lookback window for precursor identification
    'window_maxima': 20,     # Window size for finding local maxima
    'time_unit': 'minutes',  # Time unit label
    'time_unit_abbr': 'min', # Abbreviation for time units
    
    # File paths
    'signal_file_name': 'complete_buoy132_signal.npy',  # Name of the experimental signal file
    'time_file_name': 'time_array.npy',                 # Name of the time array file
    'complete_signal_dir': 'complete_signal',           # Directory containing complete signal
    'recurrence_dir': 'signal_segments_recurrence',     # Directory containing recurrence matrices
    'output_dir': 'graphical_results',                  # Directory to save analysis output
    'model_file': 'recurrence_matrix_cnn.pth',          # CNN model file name
    
    # Analysis parameters
    'confidence_bins': 21,   # Number of bins for confidence histogram (creates 20 intervals)
    'confidence_range': (0, 1),  # Range for confidence histogram
}

# =========================
# 1. CNN Model Definition
# =========================
class RecurrenceMatrixCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(RecurrenceMatrixCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# =========================
# 2. Load Trained Model
# =========================
def load_trained_model(model_path, device):
    model = RecurrenceMatrixCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# =========================
# 3. Predict from Recurrence Matrices
# =========================
def predict(model, npz_file, device):
    """Load a .npz file, extract matrix, and predict the class."""
    data = np.load(npz_file)
    matrix_key = list(data.keys())[0]
    matrix = data[matrix_key]

    matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(matrix)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    return predicted_class.item(), confidence.item()

# =========================
# 4. Load Original Signal
# =========================
def load_original_signal(script_dir):
    signal_file = script_dir / PARAMS['complete_signal_dir'] / PARAMS['signal_file_name']
    time_file = script_dir / PARAMS['complete_signal_dir'] / PARAMS['time_file_name']
    
    if not signal_file.exists() or not time_file.exists():
        raise FileNotFoundError(f"Original signal files not found in {script_dir / PARAMS['complete_signal_dir']}")
    
    signal = np.load(signal_file)
    time_array = np.load(time_file)
    
    print(f"Loaded original signal from {signal_file}")
    print(f"Signal length: {len(signal)} points")
    print(f"Time array length: {len(time_array)} points")
    print(f"Time range: {time_array[0]:.2f} to {time_array[-1]:.2f} {PARAMS['time_unit']}")
    
    return signal, time_array

# =========================
# 5. Map Segment Index to Time (With Updated Overlap)
# =========================
def map_segment_index_to_time(segment_idx):
    """
    Maps an index of a segment to its corresponding time in the original time array.
    This version handles the case with the specified overlap between segments.
    
    Parameters:
    -----------
    segment_idx : int
        Index of the segment in the ordered list of segments
    
    Returns:
    --------
    float : The corresponding time in time units for the center of the segment
    """
    # Calculate how many samples we move forward with each new segment
    step_size = PARAMS['window_size'] - PARAMS['overlap']
    
    # Calculate the start index of this segment in the original signal
    start_idx = segment_idx * step_size
    
    # Calculate the center index of this segment in the original signal
    center_idx = start_idx + (PARAMS['window_size'] // 2)
    
    # Convert to time
    return center_idx * PARAMS['dt']

# =========================
# 6. Peak Detection Functions
# =========================
def find_local_maxima(signal, window):
    """Find local maxima in the signal using a specified window"""
    local_max_indices = argrelextrema(signal, np.greater, order=window)[0]
    return local_max_indices

# =========================
# 7. Precursor Identification Approach
# =========================
def identify_precursors(signal, time_array, type1_predictions, type1_confidences):
    """
    Approach for identifying precursors:
    1. Identify all local maxima (peaks) in the signal
    2. Classify peaks as extreme or non-extreme based on threshold
    3. For each peak, find the associated Type I predictions (if any)
    4. For extreme peaks, select the leftmost Type I prediction as the precursor
    
    Returns:
        extreme_peaks: Indices of extreme peaks
        non_extreme_peaks: Indices of non-extreme peaks
        precursors: Dictionary mapping extreme peak indices to precursor times
        filtered_type1_dict: Dictionary of Type I predictions associated with extreme peaks
    """
    # 1. Find all local maxima
    all_peaks = find_local_maxima(signal, PARAMS['window_maxima'])
    print(f"Found {len(all_peaks)} local maxima in the signal")
    
    # 2. Determine threshold for extreme events
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    threshold = mean_val + PARAMS['threshold_factor'] * std_val
    print(f"Threshold for extreme events: {threshold:.4f}")
    
    # 3. Classify peaks as extreme or non-extreme
    extreme_peaks = []
    non_extreme_peaks = []
    
    for peak_idx in all_peaks:
        if signal[peak_idx] > threshold:
            extreme_peaks.append(peak_idx)
        else:
            non_extreme_peaks.append(peak_idx)
    
    extreme_peaks = np.array(extreme_peaks)
    non_extreme_peaks = np.array(non_extreme_peaks)
    
    print(f"Identified {len(extreme_peaks)} extreme peaks and {len(non_extreme_peaks)} non-extreme peaks")
    
    # 4. Create a mapping from each prediction's approximate index to its details
    pred_indices = {}
    for i, pred_time in enumerate(type1_predictions):
        pred_idx = np.argmin(np.abs(time_array - pred_time))
        pred_indices[pred_idx] = {
            'time': pred_time,
            'confidence': type1_confidences[i]
        }
    
    # 5. Associate Type I predictions with peaks
    peak_to_predictions = {}
    
    # Combine all peaks into a single list of indices
    all_peak_indices = np.concatenate([extreme_peaks, non_extreme_peaks]) if len(extreme_peaks) > 0 and len(non_extreme_peaks) > 0 else (extreme_peaks if len(extreme_peaks) > 0 else non_extreme_peaks)
    
    # For each peak, find preceding Type I predictions within the lookback window
    for peak_idx in all_peak_indices:
        # Ensure peak_idx is an integer
        peak_idx = int(peak_idx)
        # Find all Type I predictions before this peak and within max_lookback
        peak_to_predictions[peak_idx] = []
        peak_time = time_array[peak_idx]
        
        for pred_idx in pred_indices.keys():
            pred_time = pred_indices[pred_idx]['time']
            # Check if prediction is before peak and within lookback window
            if pred_time < peak_time and (peak_time - pred_time) <= (PARAMS['max_lookback'] * PARAMS['dt']):
                peak_to_predictions[peak_idx].append(pred_idx)
    
    # 6. For extreme peaks, identify the best precursor (leftmost prediction)
    precursors = {}
    for peak_idx in extreme_peaks:
        # Ensure peak_idx is an integer
        peak_idx = int(peak_idx)
        if peak_to_predictions[peak_idx]:
            # Sort the predictions by index (earliest first)
            sorted_preds = sorted(peak_to_predictions[peak_idx], 
                                 key=lambda idx: pred_indices[idx]['time'])
            leftmost_pred_idx = sorted_preds[0]
            precursors[peak_idx] = pred_indices[leftmost_pred_idx]['time']
            print(f"Extreme peak at t={time_array[peak_idx]:.2f} {PARAMS['time_unit_abbr']} (amplitude: {signal[peak_idx]:.2f})")
            print(f"  Best precursor at t={pred_indices[leftmost_pred_idx]['time']:.2f} {PARAMS['time_unit_abbr']} (leftmost prediction)")
    
    # 7. Create a filtered dictionary with only Type I predictions associated with extreme peaks
    filtered_type1_dict = {}
    for peak_idx in extreme_peaks:
        # Ensure peak_idx is an integer
        peak_idx = int(peak_idx)
        for pred_idx in peak_to_predictions[peak_idx]:
            pred_time = pred_indices[pred_idx]['time']
            pred_conf = pred_indices[pred_idx]['confidence']
            filtered_type1_dict[pred_time] = pred_conf
    
    print(f"Filtered {len(type1_predictions)} Type I predictions down to {len(filtered_type1_dict)} associated with extreme peaks")
    
    return extreme_peaks, non_extreme_peaks, precursors, filtered_type1_dict, threshold

# =========================
# 8. Confidence Histogram Analysis (New Function)
# =========================
def analyze_confidence_histogram(type1_predictions, type1_confidences, filtered_type1_dict, output_dir):
    """
    Create a histogram of confidence values for Type I (2nd quadrant) predictions.
    Enhanced with basic statistical analysis for the PDF vs Confidence graph.
    
    Parameters:
    -----------
    type1_predictions : list
        Times of all Type I (2nd quadrant) predictions
    type1_confidences : list
        List of confidence values for all Type I (2nd quadrant) predictions
    filtered_type1_dict : dict
        Dictionary of Type I predictions associated with extreme events
    output_dir : Path
        Directory to save the output data
    """
    import numpy as np
    
    # Convert to numpy arrays
    all_confidences = np.array(type1_confidences)
    
    # Create histogram data
    bins = np.linspace(PARAMS['confidence_range'][0], PARAMS['confidence_range'][1], PARAMS['confidence_bins'])
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create histogram (normalized to represent PDF)
    all_hist, _ = np.histogram(all_confidences, bins=bins, density=True)
    
    # Split into successful and unsuccessful predictions
    successful_confidences = np.array(list(filtered_type1_dict.values()))
    unsuccessful_confidences = np.array([conf for i, conf in enumerate(all_confidences) 
                                       if type1_predictions[i] not in filtered_type1_dict])
    
    if len(successful_confidences) > 0:
        succ_hist, _ = np.histogram(successful_confidences, bins=bins, density=True)
    else:
        succ_hist = np.zeros_like(all_hist)
    
    if len(unsuccessful_confidences) > 0:
        unsucc_hist, _ = np.histogram(unsuccessful_confidences, bins=bins, density=True)
    else:
        unsucc_hist = np.zeros_like(all_hist)
    
    # Calculate basic statistics
    # Mean, median, std for all confidences
    mean_all = np.mean(all_confidences) if len(all_confidences) > 0 else 0
    median_all = np.median(all_confidences) if len(all_confidences) > 0 else 0
    std_all = np.std(all_confidences) if len(all_confidences) > 0 else 0
    
    # Mean, median, std for successful predictions
    mean_succ = np.mean(successful_confidences) if len(successful_confidences) > 0 else 0
    median_succ = np.median(successful_confidences) if len(successful_confidences) > 0 else 0
    std_succ = np.std(successful_confidences) if len(successful_confidences) > 0 else 0
    
    # Mean, median, std for unsuccessful predictions
    mean_unsucc = np.mean(unsuccessful_confidences) if len(unsuccessful_confidences) > 0 else 0
    median_unsucc = np.median(unsuccessful_confidences) if len(unsuccessful_confidences) > 0 else 0
    std_unsucc = np.std(unsuccessful_confidences) if len(unsuccessful_confidences) > 0 else 0
    
    # Calculate success rate by confidence bin
    success_rate_by_bin = np.zeros(len(bin_centers))
    total_in_bin = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers)):
        lower = bins[i]
        upper = bins[i+1]
        
        # Count total predictions in this bin
        in_bin = np.sum((all_confidences >= lower) & (all_confidences < upper))
        total_in_bin[i] = in_bin
        
        # Count successful predictions in this bin
        succ_in_bin = np.sum((successful_confidences >= lower) & (successful_confidences < upper))
        
        # Calculate success rate (avoid division by zero)
        if in_bin > 0:
            success_rate_by_bin[i] = succ_in_bin / in_bin
    
    # Save the data for external plotting
    confidence_data = {
        # Basic histogram data
        'bin_centers': bin_centers,
        'bin_edges': bins,
        'all_hist': all_hist,
        'succ_hist': succ_hist,
        'unsucc_hist': unsucc_hist,
        'all_confidences': all_confidences,
        'successful_confidences': successful_confidences,
        'unsuccessful_confidences': unsuccessful_confidences,
        'total_predictions': len(all_confidences),
        'successful_predictions': len(successful_confidences),
        
        # Basic statistics
        'mean_all': mean_all,
        'median_all': median_all,
        'std_all': std_all,
        'mean_succ': mean_succ,
        'median_succ': median_succ,
        'std_succ': std_succ,
        'mean_unsucc': mean_unsucc,
        'median_unsucc': median_unsucc,
        'std_unsucc': std_unsucc,
        
        # Success rate by bin
        'success_rate_by_bin': success_rate_by_bin,
        'total_in_bin': total_in_bin,
        
        # Labels and units for plotting
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Confidence',
        'y_label': 'Probability Density'
    }
    
    np.savez(output_dir / "confidence_histogram.npz", **confidence_data)
    print(f"Saved enhanced confidence histogram data to {output_dir / 'confidence_histogram.npz'}")
    
    # Print basic statistics
    print("\n**Confidence Histogram Analysis:**")
    print(f"Total Type I (2nd quad) predictions: {len(all_confidences)}")
    print(f"Associated with extreme events: {len(successful_confidences)}")
    
    if len(all_confidences) > 0:
        print(f"Success rate: {len(successful_confidences)/len(all_confidences)*100:.1f}% of predictions")
    else:
        print("Success rate: N/A (no predictions found)")
    
    print("\n**Confidence Statistics:**")
    if len(all_confidences) > 0:
        print(f"All predictions - Mean: {mean_all:.3f}, Median: {median_all:.3f}, Std: {std_all:.3f}")
        if len(successful_confidences) > 0:
            print(f"Successful predictions - Mean: {mean_succ:.3f}, Median: {median_succ:.3f}, Std: {std_succ:.3f}")
        if len(unsuccessful_confidences) > 0:
            print(f"Unsuccessful predictions - Mean: {mean_unsucc:.3f}, Median: {median_unsucc:.3f}, Std: {std_unsucc:.3f}")
    else:
        print("No predictions found - cannot calculate statistics")
    
    # Calculate percentage of predictions in different confidence brackets
    if len(all_confidences) > 0:
        brackets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        print("\n**Confidence Brackets:**")
        for low, high in brackets:
            # All predictions in this bracket
            count = sum(1 for conf in all_confidences if low <= conf < high)
            percentage = (count / len(all_confidences)) * 100
            
            # Successful predictions in this bracket
            succ_count = sum(1 for conf in successful_confidences if low <= conf < high)
            succ_percentage = (succ_count / count) * 100 if count > 0 else 0
            
            print(f"{low:.1f}-{high:.1f}: {count} predictions ({percentage:.1f}%), Success rate: {succ_percentage:.1f}%")
    else:
        print("\n**Confidence Brackets:**")
        print("No predictions found - cannot analyze confidence brackets")

# =========================
# 9. Save Data for External Plotting
# =========================
def save_data_for_plotting(t_physical, full_signal, extreme_peaks, non_extreme_peaks, 
                          precursors, filtered_type1_dict, threshold, type1_predictions, 
                          type1_confidences, delta_t_values, output_dir):
    """
    Save all the data needed for external plotting to NPZ files.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save signal data
    signal_data = {
        'time': t_physical,
        'signal': full_signal,
        'threshold': threshold,
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Time',  # Simplified x-label as requested
        'y_label': 'Signal Amplitude'
    }
    np.savez(output_dir / "signal_data.npz", **signal_data)
    
    # 2. Save extreme and non-extreme peak data
    peak_data = {
        'extreme_indices': extreme_peaks,
        'extreme_times': t_physical[extreme_peaks] if len(extreme_peaks) > 0 else np.array([]),
        'extreme_values': full_signal[extreme_peaks] if len(extreme_peaks) > 0 else np.array([]),
        'non_extreme_indices': non_extreme_peaks,
        'non_extreme_times': t_physical[non_extreme_peaks] if len(non_extreme_peaks) > 0 else np.array([]),
        'non_extreme_values': full_signal[non_extreme_peaks] if len(non_extreme_peaks) > 0 else np.array([]),
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Time',
        'y_label': 'Signal Amplitude'
    }
    np.savez(output_dir / "peak_data.npz", **peak_data)
    
    # 3. Save precursor data
    precursor_times = np.array(list(precursors.values()))
    precursor_indices = np.array(list(precursors.keys()))
    
    # Calculate precursor values
    precursor_values = []
    for precursor_time in precursors.values():
        precursor_idx = np.argmin(np.abs(t_physical - precursor_time))
        precursor_values.append(full_signal[precursor_idx])
    
    precursor_data = {
        'peak_indices': precursor_indices,
        'precursor_times': precursor_times,
        'precursor_values': np.array(precursor_values),
        'peak_times': t_physical[precursor_indices] if len(precursor_indices) > 0 else np.array([]),
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Time',
        'y_label': 'Signal Amplitude'
    }
    np.savez(output_dir / "precursor_data.npz", **precursor_data)
    
    # 4. Save time difference data
    warning_time_data = {
        'delta_t_values': np.array(delta_t_values),
        'mean_dt': np.mean(delta_t_values) if delta_t_values else 0,
        'median_dt': np.median(delta_t_values) if delta_t_values else 0,
        'std_dt': np.std(delta_t_values) if delta_t_values else 0,
        'min_dt': np.min(delta_t_values) if delta_t_values else 0,
        'max_dt': np.max(delta_t_values) if delta_t_values else 0,
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Warning Time',
        'y_label': 'Probability Density'
    }
    np.savez(output_dir / "warning_time_data.npz", **warning_time_data)
    
    # 5. Save prediction data
    prediction_data = {
        'type1_times': np.array(type1_predictions),
        'type1_confidences': np.array(type1_confidences),
        'filtered_type1_times': np.array(list(filtered_type1_dict.keys())),
        'filtered_type1_confidences': np.array(list(filtered_type1_dict.values())),
        'time_unit': PARAMS['time_unit'],
        'time_unit_abbr': PARAMS['time_unit_abbr'],
        'x_label': 'Time',
        'y_label': 'Confidence'
    }
    np.savez(output_dir / "prediction_data.npz", **prediction_data)
    
    print(f"Saved signal data to {output_dir / 'signal_data.npz'}")
    print(f"Saved peak data to {output_dir / 'peak_data.npz'}")
    print(f"Saved precursor data to {output_dir / 'precursor_data.npz'}")
    print(f"Saved warning time data to {output_dir / 'warning_time_data.npz'}")
    print(f"Saved prediction data to {output_dir / 'prediction_data.npz'}")

# =========================
# 10. Main Execution
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = Path(__file__).parent
    model_path = script_dir / PARAMS['model_file']
    output_dir = script_dir / PARAMS['output_dir']

    # Check if model exists
    if not model_path.exists():
        print(f"Warning: Model file not found at {model_path}")
        print("Please ensure the model file is correctly placed.")
    else:
        model = load_trained_model(model_path, device)
        print("Model loaded successfully.")

    recurrence_dir = script_dir / PARAMS['recurrence_dir']
    
    # Check if recurrence directory exists
    if not recurrence_dir.exists():
        print(f"Warning: Recurrence matrices directory not found at {recurrence_dir}")
        print("Creating directory in case it needs to be populated later")
        recurrence_dir.mkdir(exist_ok=True)

    # Load the original complete signal and time array
    print("\nLoading original complete signal...")
    try:
        full_signal, t_physical = load_original_signal(script_dir)
        # Set the time span dynamically based on loaded data
        PARAMS['t_span'] = (t_physical[0], t_physical[-1])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the experimental data analysis script to generate the signal files before running this code.")
        exit(1)
    
    # Get CNN Predictions from recurrence matrices
    type1_predictions = []  # Will store times of Type I (2nd quadrant) predictions
    type1_confidences = []  # Will store confidences of Type I predictions
    type2_predictions = []  # Will store times of Type I (4th quadrant) predictions
    type2_confidences = []  # Will store confidences of Type I (4th quadrant) predictions
    
    # Check if there are any NPZ files in the recurrence directory
    recurrence_files = list(sorted(recurrence_dir.glob("segment_*.npz")))
    if not recurrence_files:
        print(f"Warning: No recurrence matrix files found in {recurrence_dir}")
        print("You may need to generate the recurrence matrices first.")
    else:
        print(f"\nProcessing {len(recurrence_files)} recurrence matrix files...")
        print("\n**Prediction Results:**")
        
        for idx, npz_file in enumerate(tqdm(recurrence_files, desc="Processing recurrence matrices")):
            try:
                predicted_class, confidence = predict(model, npz_file, device)
                
                class_desc = {
                    0: "Type 0 (K <= 0.2)",
                    1: "Type 1 (Type I, 2nd quadrant)",
                    2: "Type 2 (Type I, 4th quadrant)",
                    3: "Type 3 (K >= 0.8)"
                }.get(predicted_class, f"Unknown Type {predicted_class}")
                
                # Map segment index to time in the original signal
                segment_time = map_segment_index_to_time(idx)
                
                # Ensure time is within the valid range
                if segment_time > t_physical[-1]:
                    segment_time = t_physical[-1]
                
                print(f"File: {npz_file.name} -> Predicted: {class_desc} (Confidence: {confidence * 100:.2f}%) at time {segment_time:.2f} {PARAMS['time_unit_abbr']}")

                # Store Type 1 and Type 2 predictions
                if predicted_class == 1:  # Type I, 2nd quadrant
                    type1_predictions.append(segment_time)
                    type1_confidences.append(confidence)
                elif predicted_class == 2:  # Type I, 4th quadrant
                    type2_predictions.append(segment_time)
                    type2_confidences.append(confidence)
            except Exception as e:
                print(f"Error processing file {npz_file}: {e}")
        
        print(f"Found {len(type1_predictions)} Type I (2nd quadrant) instances")
        print(f"Found {len(type2_predictions)} Type I (4th quadrant) instances")

        # Identify precursors
        print("\n**Identifying Precursors:**")
        
        extreme_peaks, non_extreme_peaks, precursors, filtered_type1_dict, threshold = identify_precursors(
            full_signal, t_physical, type1_predictions, type1_confidences
        )
        
        # Convert dictionary keys to list for data saving
        filtered_type1_predictions = list(filtered_type1_dict.keys())
        filtered_type1_confidences = list(filtered_type1_dict.values())
        
        # Get precursor times as list
        precursor_times = list(precursors.values())
        
        # Calculate time differences (delta_t) between extreme events and their precursors
        delta_t_values = []
        for peak_idx, precursor_time in precursors.items():
            peak_time = t_physical[peak_idx]
            delta_t = peak_time - precursor_time
            delta_t_values.append(delta_t)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze confidence histogram (New function call)
        print("\n**Analyzing Confidence Histogram:**")
        analyze_confidence_histogram(type1_predictions, type1_confidences, filtered_type1_dict, output_dir)
        
        # Save data for external plotting instead of creating visualizations
        save_data_for_plotting(
            t_physical, full_signal, extreme_peaks, non_extreme_peaks, 
            precursors, filtered_type1_dict, threshold, type1_predictions, 
            type1_confidences, delta_t_values, output_dir
        )
        
        # Save type2 prediction data for completeness
        type2_data = {
            'type2_times': np.array(type2_predictions),
            'type2_confidences': np.array(type2_confidences),
            'time_unit': PARAMS['time_unit'],
            'time_unit_abbr': PARAMS['time_unit_abbr'],
            'x_label': 'Time',
            'y_label': 'Confidence'
        }
        np.savez(output_dir / "type2_prediction_data.npz", **type2_data)
        print(f"Saved Type II prediction data to {output_dir / 'type2_prediction_data.npz'}")
        
        # Generate Statistics
        print("\n**Statistics:**")
        print(f"Total Type I (2nd quad) predictions: {len(type1_predictions)}")
        print(f"Total Type I (4th quad) predictions: {len(type2_predictions)}")
        
        # Handle division by zero scenarios
        if type1_predictions:
            type1_percentage = len(filtered_type1_predictions)/len(type1_predictions)*100
            print(f"Type I (2nd quad) predictions associated with extreme peaks: {len(filtered_type1_predictions)} ({type1_percentage:.1f}%)")
        else:
            print("No Type I (2nd quad) predictions found")
        
        if extreme_peaks.size > 0:
            precursor_percentage = len(precursors)/len(extreme_peaks)*100
            print(f"Extreme peaks with identified precursors: {len(precursors)}/{len(extreme_peaks)} ({precursor_percentage:.1f}%)")
        else:
            print("No extreme peaks detected")
        
        # Add statistics about time differences
        if delta_t_values:
            mean_dt = np.mean(delta_t_values)
            median_dt = np.median(delta_t_values)
            std_dt = np.std(delta_t_values)
            
            print("\n**Time Difference Statistics:**")
            print(f"Average warning time: {mean_dt:.2f} {PARAMS['time_unit']}")
            print(f"Median warning time: {median_dt:.2f} {PARAMS['time_unit']}")
            print(f"Standard deviation of warning times: {std_dt:.2f} {PARAMS['time_unit']}")
            print(f"Minimum warning time: {min(delta_t_values):.2f} {PARAMS['time_unit']}")
            print(f"Maximum warning time: {max(delta_t_values):.2f} {PARAMS['time_unit']}")
        else:
            print("\n**Time Difference Statistics:**")
            print("No precursors identified for extreme peaks. No warning time statistics available.")
        
        # Save statistics to a text file
        stats_file = output_dir / "statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("**Statistics:**\n")
            f.write(f"Total Type I (2nd quad) predictions: {len(type1_predictions)}\n")
            f.write(f"Total Type I (4th quad) predictions: {len(type2_predictions)}\n")
            
            if type1_predictions:
                f.write(f"Type I (2nd quad) predictions associated with extreme peaks: {len(filtered_type1_predictions)} ({type1_percentage:.1f}%)\n")
            else:
                f.write("No Type I (2nd quad) predictions found\n")
            
            if extreme_peaks.size > 0:
                f.write(f"Extreme peaks with identified precursors: {len(precursors)}/{len(extreme_peaks)} ({precursor_percentage:.1f}%)\n")
            else:
                f.write("No extreme peaks detected\n")
            
            if delta_t_values:
                f.write("\n**Time Difference Statistics:**\n")
                f.write(f"Average warning time: {mean_dt:.2f} {PARAMS['time_unit']}\n")
                f.write(f"Median warning time: {median_dt:.2f} {PARAMS['time_unit']}\n")
                f.write(f"Standard deviation of warning times: {std_dt:.2f} {PARAMS['time_unit']}\n")
                f.write(f"Minimum warning time: {min(delta_t_values):.2f} {PARAMS['time_unit']}\n")
                f.write(f"Maximum warning time: {max(delta_t_values):.2f} {PARAMS['time_unit']}\n")
            else:
                f.write("\n**Time Difference Statistics:**\n")
                f.write("No precursors identified for extreme peaks. No warning time statistics available.\n")
                
            # Add confidence statistics to the statistics file
            if len(type1_confidences) > 0:
                f.write("\n**Confidence Statistics:**\n")
                
                # Calculate basic statistics for all confidences
                mean_all = np.mean(type1_confidences)
                median_all = np.median(type1_confidences)
                std_all = np.std(type1_confidences)
                f.write(f"All predictions - Mean: {mean_all:.3f}, Median: {median_all:.3f}, Std: {std_all:.3f}\n")
                
                # Calculate statistics for successful confidences
                if filtered_type1_confidences:
                    mean_succ = np.mean(filtered_type1_confidences)
                    median_succ = np.median(filtered_type1_confidences)
                    std_succ = np.std(filtered_type1_confidences)
                    f.write(f"Successful predictions - Mean: {mean_succ:.3f}, Median: {median_succ:.3f}, Std: {std_succ:.3f}\n")
                
                # Calculate statistics for unsuccessful confidences
                unsuccessful_confidences = [conf for i, conf in enumerate(type1_confidences) 
                                          if type1_predictions[i] not in filtered_type1_dict]
                if unsuccessful_confidences:
                    mean_unsucc = np.mean(unsuccessful_confidences)
                    median_unsucc = np.median(unsuccessful_confidences)
                    std_unsucc = np.std(unsuccessful_confidences)
                    f.write(f"Unsuccessful predictions - Mean: {mean_unsucc:.3f}, Median: {median_unsucc:.3f}, Std: {std_unsucc:.3f}\n")
                
                # Calculate percentage of predictions in different confidence brackets
                brackets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                
                f.write("\n**Confidence Brackets:**\n")
                for low, high in brackets:
                    # All predictions in this bracket
                    count = sum(1 for conf in type1_confidences if low <= conf < high)
                    percentage = (count / len(type1_confidences)) * 100
                    
                    # Successful predictions in this bracket
                    succ_count = sum(1 for conf in filtered_type1_confidences if low <= conf < high)
                    succ_percentage = (succ_count / count) * 100 if count > 0 else 0
                    
                    f.write(f"{low:.1f}-{high:.1f}: {count} predictions ({percentage:.1f}%), Success rate: {succ_percentage:.1f}%\n")
        
        print(f"Saved statistics to {stats_file}")