import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import hilbert
from tqdm import tqdm

def apply_hilbert_envelope(signal):
    """Apply Hilbert transform to extract amplitude envelope"""
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    return envelope

def apply_square_transformation(signal):
    """Apply cubic transformation to enhance extreme events"""
    return np.sign(signal) * (np.abs(signal)**3)

def detect_extreme_peaks(signal, threshold_factor=4.0):
    """Detect extreme peaks using the Peaks Over Threshold (POT) method"""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    threshold = mean_val + threshold_factor * std_val
    extreme_indices = np.where(signal > threshold)[0]
    return extreme_indices, threshold

def segment_and_save_signal(t, signal, window_size=500, overlap=250):
    """
    Segment the signal using overlapping windows and save to files
    
    Args:
        t: Time array
        signal: Signal array to segment
        window_size: Number of points in each window
        overlap: Number of points to overlap between consecutive windows
    """
    
    # Create directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_segments")
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate step size (with overlap)
    step = window_size - overlap
    
    # Check if step size is valid
    if step <= 0:
        print("Error: Overlap must be less than window size")
        return
    
    # Calculate number of segments
    n_segments = 1 + (len(signal) - window_size) // step
    
    print(f"Creating {n_segments} segments of size {window_size} with overlap of {overlap} points")
    
    # Generate segments
    for i in range(n_segments):
        start_idx = i * step
        end_idx = start_idx + window_size
        
        # Make sure we don't go beyond the signal length
        if end_idx > len(signal):
            break
        
        # Extract segment data
        t_segment = t[start_idx:end_idx]
        signal_segment = signal[start_idx:end_idx]
        
        # Save segment as .npy file with simplified name
        filename = os.path.join(save_dir, f"segment_{i+1:03d}.npy")
        
        # Explicitly save with .npy extension only
        # Use allow_pickle=False to ensure clean numpy format
        np.save(filename, signal_segment, allow_pickle=False)
        
        # Verify the file was saved correctly
        if not os.path.exists(filename):
            print(f"Warning: Failed to save {filename}")
    
    # Verify all files have been saved with correct extension
    saved_files = [f for f in os.listdir(save_dir) if f.startswith("segment_") and f.endswith(".npy")]
    print(f"Saved {len(saved_files)} segments in {save_dir}")
    
    # Check if there are any incorrect extension files
    incorrect_files = [f for f in os.listdir(save_dir) if f.startswith("segment_") and not f.endswith(".npy")]
    if incorrect_files:
        print(f"Warning: Found {len(incorrect_files)} files with incorrect extensions.")
        print("You may want to manually remove:", incorrect_files[:5], "..." if len(incorrect_files) > 5 else "")
    
    print(f"These segments can be concatenated with appropriate handling of the overlapping regions")

def save_complete_signal(T, signal):
    """
    Save the complete signal to a separate folder.
    This is the original signal without any segmentation or overlap.
    
    Parameters:
    T : array
        Time array
    signal : array
        Signal array to save
    """
    # Create directory if it doesn't exist
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "complete_signal")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save signal
    complete_signal = signal
    
    # Save as .npy file
    np.save(os.path.join(save_dir, "complete_buoy132_signal.npy"), complete_signal)
    
    # Save time array separately
    np.save(os.path.join(save_dir, "time_array.npy"), T)
    
    print(f"Saved complete signal (length: {len(complete_signal)}) in {save_dir}")
    print(f"Time range: {T[0]:.2f} to {T[-1]:.2f}")

# Main script
print("Starting experimental data analysis...")

# Load the specific .npz file
data = np.load('RWs_H_g_2p2_tadv_1min_buoy_132.npz')

# List all arrays stored in the .npz file
print("Keys in the .npz file:", data.files)

# Just print the column names - no detailed exploration
print("\nAvailable arrays (columns) in the data file:")
for i, key in enumerate(data.files, 1):
    array = data[key]
    print(f"{i}. '{key}' - Shape: {array.shape}")

# Pick one waveform sample (squeeze to remove the last singleton dimension)
original_signal = data['wave_data_train'][0].squeeze()

print(f"\nSelected experimental signal shape: {original_signal.shape}")

# Apply transformations step by step
hilbert_envelope = apply_hilbert_envelope(original_signal)
square_hilbert = apply_square_transformation(hilbert_envelope)

print(f"Applied Hilbert envelope transformation")
print(f"Original signal range: {np.min(original_signal):.4f} to {np.max(original_signal):.4f}")
print(f"Hilbert envelope range: {np.min(hilbert_envelope):.4f} to {np.max(hilbert_envelope):.4f}")
print(f"Applied square transform to Hilbert envelope")
print(f"Square(Hilbert envelope) range: {np.min(square_hilbert):.4f} to {np.max(square_hilbert):.4f}")

# Create time array based on signal length
# Assuming 1-minute sampling as indicated in filename
dt = 1.0  # 1 minute per sample, adjust if needed
T = np.arange(0, len(original_signal) * dt, dt)

print(f"Signal length: {len(original_signal)}")
print(f"Time range: {T[0]:.2f} to {T[-1]:.2f} minutes")

# Save complete signal (using the final transformed signal)
save_complete_signal(T, square_hilbert)

# Create comprehensive visualization with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# 1. Original Signal
axes[0].plot(T, original_signal, color='blue', linewidth=1, label='Original Wave Data')
axes[0].set_title("Original Wave Data (Buoy 132)", fontsize=14)
axes[0].set_ylabel("Amplitude", fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Hilbert Envelope
axes[1].plot(T, original_signal, color='lightblue', alpha=0.6, linewidth=0.8, label='Original Signal')
axes[1].plot(T, hilbert_envelope, color='darkgreen', linewidth=2, label='Hilbert Envelope')
axes[1].plot(T, -hilbert_envelope, color='darkgreen', linewidth=2, alpha=0.7)
axes[1].set_title("Hilbert Envelope", fontsize=14)
axes[1].set_ylabel("Envelope Amplitude", fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Square Transform of Hilbert Envelope
axes[2].plot(T, square_hilbert, color='purple', linewidth=1, label='Square(Hilbert Envelope)')
axes[2].set_title("Square Transform of Hilbert Envelope (Enhanced)", fontsize=14)
axes[2].set_ylabel("Square(Envelope) Amplitude", fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Extreme Events Detection on Square(Hilbert Envelope)
extreme_indices, threshold = detect_extreme_peaks(square_hilbert, threshold_factor=2.0)

axes[3].plot(T, square_hilbert, color='purple', linewidth=1, label='Square(Hilbert Envelope)')
axes[3].axhline(y=threshold, color='r', linestyle='--', 
               label=f'Threshold ({threshold:.4f})')
axes[3].scatter(T[extreme_indices], square_hilbert[extreme_indices], color='red', s=30,
               label=f'Extreme Events ({len(extreme_indices)})')
axes[3].set_title("Extreme Events Detection in Square(Hilbert Envelope)", fontsize=14)
axes[3].set_xlabel("Time (minutes)", fontsize=12)
axes[3].set_ylabel("Square(Envelope) Amplitude", fontsize=12)
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()

# Display the figure
plt.show()

# Save the figure
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, 'extreme_events_hilbert_square_analysis_buoy132.png')
plt.savefig(file_path, dpi=300)

print(f"Extreme events analysis complete. Figure saved to: {file_path}")

# Segment and save the final transformed signal
window_size = 150  # Number of points in each window
overlap = 135      # Number of points to overlap between consecutive windows

print(f"\nSegmenting Square(Hilbert Envelope) signal...")
segment_and_save_signal(T, square_hilbert, window_size, overlap)

print("\nAnalysis complete!")
print("Data saved in:")
print("- complete_signal/ (Square(Hilbert Envelope) signal)")  
print("- signal_segments/ (Square(Hilbert Envelope) sliding windows)")

# Print comprehensive transformation statistics
print(f"\n" + "="*60)
print("TRANSFORMATION ANALYSIS")
print("="*60)

print(f"\nORIGINAL SIGNAL:")
print(f"  Range: {np.min(original_signal):.4f} to {np.max(original_signal):.4f}")
print(f"  Mean: {np.mean(original_signal):.4f}")
print(f"  Std: {np.std(original_signal):.4f}")

print(f"\nHILBERT ENVELOPE:")
print(f"  Range: {np.min(hilbert_envelope):.4f} to {np.max(hilbert_envelope):.4f}")
print(f"  Mean: {np.mean(hilbert_envelope):.4f}")
print(f"  Std: {np.std(hilbert_envelope):.4f}")

print(f"\nSQUARE(HILBERT ENVELOPE):")
print(f"  Range: {np.min(square_hilbert):.4f} to {np.max(square_hilbert):.4f}")
print(f"  Mean: {np.mean(square_hilbert):.4f}")
print(f"  Std: {np.std(square_hilbert):.4f}")
print(f"  Extreme events detected: {len(extreme_indices)}")

# Peak enhancement analysis
peak_idx = np.argmax(np.abs(original_signal))
original_peak = np.abs(original_signal[peak_idx])
hilbert_peak = hilbert_envelope[peak_idx]
square_peak = square_hilbert[peak_idx]

print(f"\nPEAK ENHANCEMENT ANALYSIS (at timestep {peak_idx}):")
print(f"  Original amplitude: {original_peak:.4f}")
print(f"  Hilbert envelope: {hilbert_peak:.4f} (factor: {hilbert_peak/original_peak:.2f}x)")
print(f"  Square enhanced: {square_peak:.4f} (factor: {square_peak/original_peak:.2f}x)")

if len(extreme_indices) > 0:
    print(f"\nEXTREME EVENTS SUMMARY:")
    print(f"  Number of extreme events: {len(extreme_indices)}")
    print(f"  Max extreme event amplitude: {np.max(square_hilbert[extreme_indices]):.4f}")
    print(f"  Extreme event locations (timesteps): {extreme_indices[:10]}{'...' if len(extreme_indices) > 10 else ''}")

print(f"\n" + "="*60)
print("The Square(Hilbert Envelope) method provides maximum enhancement")
print("by combining amplitude extraction with nonlinear amplification!")
print("="*60)