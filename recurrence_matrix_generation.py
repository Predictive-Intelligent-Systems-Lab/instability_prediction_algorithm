import numpy as np
import os
from pathlib import Path
import re
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def time_delay_embedding(data, embed_dim, tau):
    """Perform time-delay embedding on the input data."""
    N = len(data) - (embed_dim - 1) * tau
    if N <= 0:
        raise ValueError("Data length is too short for the given embedding parameters")
    
    embedded = np.zeros((N, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = data[i * tau:i * tau + N]
    return embedded

def calculate_recurrence_matrix(embedded_data, threshold=None):
    """Calculate the recurrence matrix using embedded data."""
    dist_matrix = np.zeros((len(embedded_data), len(embedded_data)))
    for i in range(len(embedded_data)):
        dist_matrix[i] = np.sqrt(np.sum((embedded_data - embedded_data[i])**2, axis=1))
    
    if threshold is None:
        threshold = 0.1 * np.max(dist_matrix)
    
    recurrence_matrix = (dist_matrix <= threshold).astype(int)
    return recurrence_matrix

def resize_matrix(matrix, size=(450, 450)):
    """Resize the matrix to the given size using interpolation."""
    zoom_factors = (size[0] / matrix.shape[0], size[1] / matrix.shape[1])
    resized_matrix = zoom(matrix, zoom_factors, order=1)  # Linear interpolation
    return resized_matrix

def parse_input_filename(filename):
    """Parse parameters from input filename."""
    # Updated pattern to match: U0_7.00_segment_0000_embed_3_tau_5.npy
    pattern = r'U0_(\d+\.\d+)_segment_(\d+)_embed_(\d+)_tau_(\d+)\.npy'
    match = re.match(pattern, filename)
    if match:
        return {
            'velocity': float(match.group(1)),
            'segment': int(match.group(2)),
            'embed_dim': int(match.group(3)),
            'tau': int(match.group(4))
        }
    return None

def parse_output_filename(filename):
    """Parse parameters from the recurrence matrix filename."""
    # Updated pattern to match: recurrence_matrix_U0_7.00_segment_0000.npy
    pattern = r'recurrence_matrix_U0_(\d+\.\d+)_segment_(\d+)\.npy'
    match = re.match(pattern, filename)
    if match:
        return {
            'velocity': float(match.group(1)),
            'segment': int(match.group(2))
        }
    return None

def process_files(input_dir, output_dir, matrix_size=(450, 450)):
    """Process all .npy files and save recurrence matrices."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve input and output directories relative to script
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(script_dir, input_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    input_files = list(Path(input_dir).glob('*.npy'))
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(input_files)} files to process")
    
    for file_path in tqdm(input_files, desc="Processing files"):
        try:
            params = parse_input_filename(file_path.name)
            if params is None:
                print(f"Skipping file with invalid name format: {file_path.name}")
                continue
            
            time_series = np.load(file_path)
            scaler = MinMaxScaler()
            time_series_normalized = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
            
            embed_dim = params['embed_dim']
            tau = params['tau']
            
            embedded_data = time_delay_embedding(
                time_series_normalized,
                embed_dim,
                tau
            )
            
            rec_matrix = calculate_recurrence_matrix(embedded_data)
            resized_rec_matrix = resize_matrix(rec_matrix, size=matrix_size)
            
            # Updated output filename format without stability info
            output_filename = f"recurrence_matrix_U0_{params['velocity']:.2f}_segment_{params['segment']:04d}.npy"
            output_path = Path(output_dir) / output_filename
            
            np.save(output_path, resized_rec_matrix)
            
        except Exception as e:
            print(f"Error processing file {file_path.name}: {str(e)}")

def visualize_first_matrices(matrix_dir, n_matrices=50):
    """Visualize first 50 recurrence matrices in a grid layout."""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve matrix directory relative to script
    if not os.path.isabs(matrix_dir):
        matrix_dir = os.path.join(script_dir, matrix_dir)
    
    all_files = sorted(list(Path(matrix_dir).glob('*.npy')))[:n_matrices]
    
    if not all_files:
        print(f"No matrix files found in {matrix_dir}")
        return
    
    fig, axes = plt.subplots(5, 10, figsize=(25, 12))
    axes_flat = axes.flatten()
    
    for i, file_path in enumerate(all_files):
        matrix = np.load(file_path)
        params = parse_output_filename(file_path.name)
        
        if params is None:
            print(f"Skipping visualization for file with invalid name: {file_path.name}")
            continue
        
        # Get matrix dimensions
        n_points = matrix.shape[0]
        
        # Plot matrix with axis values
        im = axes_flat[i].imshow(matrix, cmap='binary', aspect='equal')
        # Updated title without stability info
        axes_flat[i].set_title(f'U={params["velocity"]}', 
                             fontsize=8, pad=3)
        
        # Set axis ticks and labels
        tick_positions = np.linspace(0, n_points-1, 5).astype(int)  # 5 ticks on each axis
        axes_flat[i].set_xticks(tick_positions)
        axes_flat[i].set_yticks(tick_positions)
        axes_flat[i].set_xticklabels(tick_positions, fontsize=6)
        axes_flat[i].set_yticklabels(tick_positions, fontsize=6)
        
        # Rotate x-axis labels for better readability
        axes_flat[i].tick_params(axis='x', rotation=45)
        
        # Add grid for better readability
        axes_flat[i].grid(False)
    
    plt.tight_layout()
    output_path = os.path.join(script_dir, 'first_50_recurrence_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set default paths relative to script location
    input_dir = os.path.join(script_dir, "data_with_embed_tau")
    output_dir = os.path.join(script_dir, "recurrence_matrices")
    
    print("Calculating and resizing recurrence matrices to 450x450...")
    process_files(input_dir, output_dir, matrix_size=(450, 450))
    print("Processing complete!")
    
    print("Generating visualization...")
    visualize_first_matrices(output_dir, n_matrices=50)
    print("Done!")