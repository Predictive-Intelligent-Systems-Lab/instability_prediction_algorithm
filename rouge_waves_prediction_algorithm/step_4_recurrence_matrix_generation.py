import numpy as np
import os
import re
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from scipy.ndimage import zoom

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

class RecurrencePlotProcessor:
    def __init__(self, matrix_size=(450, 450)):
        """Initialize processor with input/output folders and matrix size."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.join(script_dir, "signal_segments_embed_tau")
        self.output_folder = os.path.join(script_dir, "signal_segments_recurrence")
        self.matrix_size = matrix_size

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def process_files(self):
        """Process all .npz files and save recurrence matrices."""
        print("\nCalculating and resizing recurrence matrices...")

        input_files = sorted([f for f in os.listdir(self.input_folder) if f.endswith('.npz')])

        if not input_files:
            print(f"No .npz files found in {self.input_folder}")
            return

        for file_name in tqdm(input_files):
            try:
                file_path = os.path.join(self.input_folder, file_name)
                data = np.load(file_path)
                signal = data['signal']
                embedding_dim = data['embedding_dim']
                optimal_tau = data['optimal_tau']

                # Normalize the signal
                scaler = MinMaxScaler()
                signal_normalized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()

                # Create embedding
                embedded_data = time_delay_embedding(signal_normalized, int(embedding_dim), int(optimal_tau))

                # Calculate and resize recurrence matrix
                rec_matrix = calculate_recurrence_matrix(embedded_data)
                resized_rec_matrix = resize_matrix(rec_matrix, size=self.matrix_size)

                # Extract segment number from the filename
                segment_match = re.search(r'_(\d+)\.', file_name)
                if segment_match:
                    segment_num = segment_match.group(1)
                else:
                    # Fallback if no segment number is found
                    segment_num = str(input_files.index(file_name) + 1).zfill(3)
                
                # Create new filename with just "segment_NUMBER.npz"
                output_filename = f"segment_{segment_num}.npz"
                output_path = os.path.join(self.output_folder, output_filename)
                
                # Save the recurrence matrix
                np.savez(output_path, recurrence_matrix=resized_rec_matrix, embedding_dim=embedding_dim, optimal_tau=optimal_tau)

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def print_summary(self):
        """Print summary of processed matrices."""
        print("\nProcessed Files Summary:")
        n_files = len([f for f in os.listdir(self.output_folder) if f.endswith('.npz')])
        print(f"Total matrices processed: {n_files}")
        print(f"Files saved in: {self.output_folder}")


# Example usage
if __name__ == "__main__":
    processor = RecurrencePlotProcessor()
    processor.process_files()
    processor.print_summary()