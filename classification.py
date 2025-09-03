import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shutil
import re

class ZeroOneTest:
    """
    Full implementation of 0-1 test for chaos detection without mathematical reduction.
    This is used for all non-Type I cases.
    """
    def __init__(self, num_c=100, c_bounds=(np.pi/5, 4*np.pi/5), threshold=0.5):
        self.num_c = num_c
        self.c_bounds = c_bounds
        self.threshold = threshold
    
    def run(self, timeseries):
        """
        Execute 0-1 test on input time series without any mathematical simplification.
        
        Args:
            timeseries (np.ndarray): Input signal
            
        Returns:
            float: K value (0 = regular, 1 = chaotic)
        """
        c_values = np.random.uniform(self.c_bounds[0], self.c_bounds[1], self.num_c)
        K_values = []
        
        for c in c_values:
            # Compute translation variables p, q
            p, q = self.compute_translations(timeseries, c)
            
            # Compute mean square displacement
            n_cut = len(timeseries) // 10
            D = self.compute_displacement(p, q, n_cut)
            
            # Compute K value
            K_values.append(self.compute_K(D))
        
        # Take median K value for robustness
        return float(np.median(K_values))
    
    def compute_translations(self, timeseries, c):
        """Compute translation variables p and q."""
        N = len(timeseries)
        mean = np.mean(timeseries)
        
        p = np.zeros(N)
        q = np.zeros(N)
        
        for n in range(1, N + 1):
            j_values = np.arange(n)
            p[n - 1] = np.sum((timeseries[:n] - mean) * np.cos(j_values * c))
            q[n - 1] = np.sum((timeseries[:n] - mean) * np.sin(j_values * c))
            
        return p, q
    
    def compute_displacement(self, p, q, n_cut):
        """Compute mean square displacement."""
        N = len(p)
        D = np.zeros(n_cut)
        
        for n in range(1, n_cut + 1):
            squared_displacements = [
                (p[j + n] - p[j]) ** 2 + (q[j + n] - q[j]) ** 2
                for j in range(N - n)
            ]
            D[n - 1] = np.mean(squared_displacements)
            
        return D
    
    def compute_K(self, D):
        """Compute correlation coefficient K."""
        n = len(D)
        time_indices = np.arange(1, n + 1)
        
        # Compute means
        mean_D = np.mean(D)
        mean_t = np.mean(time_indices)
        
        # Center variables
        D_centered = D - mean_D
        t_centered = time_indices - mean_t
        
        # Compute correlation coefficient
        covariance = np.mean(D_centered * t_centered)
        var_D = np.mean(D_centered**2)
        var_t = np.mean(t_centered**2)
        
        # Return normalized correlation
        if var_D * var_t > 0:
            return min(covariance / np.sqrt(var_D * var_t), 1.0)
        return 0.0

class RecurrenceMatricesAnalyzer:
    """
    Class for analyzing recurrence matrices and applying the updated classification:
    1. If Type I intermittency is detected (F_b ≈ 1), check the centroid position:
       - If centroid is in 2nd quadrant (top left), confirm as Type I, label as 1
       - If centroid is in 4th quadrant (bottom right), label as 2
    2. If K <= 0.2, label as 0
    3. If K >= 0.8, label as 3
    4. If none of these are detected, don't save the matrix (type_code = -1)
    """
    
    def __init__(self, matrices_folder=None, output_folder=None):
        """
        Initialize the analyzer with folders for matrices and output.
        """
        # Get the directory where this script is located
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths relative to script location
        if matrices_folder is None:
            self.matrices_folder = script_dir / 'recurrence_matrices'
        else:
            self.matrices_folder = Path(matrices_folder)
            
        if output_folder is None:
            self.output_folder = script_dir / 'classified_matrices'
        else:
            self.output_folder = Path(output_folder)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
            
        # Initialize empty lists for storing data
        self.matrix_files = []
        self.results = []
        
        # Initialize 0-1 test
        self.zero_one_test = ZeroOneTest()
        
        # Path to the original time series segments
        self.segments_dir = script_dir / "pressure_segments"
        
        # Print the detected paths for verification
        print(f"Matrices folder: {self.matrices_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Segments folder: {self.segments_dir}")
        
    def find_matrix_files(self):
        """
        Find all .npy files in the matrices folder.
        """
        try:
            # Check if the matrices folder exists
            if not self.matrices_folder.exists():
                print(f"Warning: Matrices folder '{self.matrices_folder}' does not exist!")
                return []
                
            # Find all .npy files in the folder
            self.matrix_files = sorted(list(self.matrices_folder.glob('*.npy')))
            print(f"Found {len(self.matrix_files)} recurrence matrix files.")
            return self.matrix_files
        except Exception as e:
            print(f"Error finding matrix files: {e}")
            return []
    
    def calculate_centroid(self, matrix, black_points):
        """
        Calculate the centroid of black points in the recurrence matrix.
        
        Args:
            matrix (np.ndarray): The recurrence matrix
            black_points (list): List of (i,j) coordinates of black points
            
        Returns:
            tuple: (centroid_i, centroid_j), quadrant (1, 2, 3, or 4)
        """
        if not black_points:
            return None, 0
            
        # Calculate centroid as the mean position of all black points
        i_coords = [i for i, j in black_points]
        j_coords = [j for i, j in black_points]
        
        centroid_i = np.mean(i_coords)
        centroid_j = np.mean(j_coords)
        
        # Determine the quadrant of the centroid
        n = matrix.shape[0]
        center_i = n / 2
        center_j = n / 2
        
        # Define quadrants:
        # 1: Top right (i < center_i, j >= center_j)
        # 2: Top left (i < center_i, j < center_j)
        # 3: Bottom left (i >= center_i, j < center_j)
        # 4: Bottom right (i >= center_i, j >= center_j)
        
        if centroid_i < center_i and centroid_j >= center_j:
            quadrant = 1  # Top right
        elif centroid_i < center_i and centroid_j < center_j:
            quadrant = 2  # Top left
        elif centroid_i >= center_i and centroid_j < center_j:
            quadrant = 3  # Bottom left
        else:  # centroid_i >= center_i and centroid_j >= center_j
            quadrant = 4  # Bottom right
            
        return (centroid_i, centroid_j), quadrant
    
    def analyze_matrix(self, matrix_path):
        """
        Apply updated classification scheme with centroid check:
        1. If Type I intermittency is detected (F_b ≈ 1), check centroid:
           - If centroid is in 2nd quadrant, confirm as Type I, label as 1
           - If centroid is in 4th quadrant, label as 2
        2. If K <= 0.2, label as 0
        3. If K >= 0.8, label as 3
        4. If none of these are detected, don't save the matrix (type_code = -1)
        """
        try:
            # Load the recurrence matrix
            matrix = np.load(matrix_path)
            
            # Extract original time series filename from matrix filename
            # Format: recurrence_matrix_U0_7.00_segment_0000.npy
            match = re.match(r'recurrence_matrix_U0_(\d+\.\d+)_segment_(\d+)\.npy', matrix_path.name)
            if match:
                velocity = float(match.group(1))
                segment_idx = int(match.group(2))
            else:
                velocity = 0.0
                segment_idx = 0
            
            # Get basic properties
            n = matrix.shape[0]  # Size of the matrix
            filename = matrix_path.name  # Filename for reference
            
            # Initialize parameters for finding the largest continuous black region
            max_region_size = 0
            max_region_coords = None
            visited = np.zeros_like(matrix)  # Keep track of visited cells
            
            # Find largest continuous black region using a flood-fill algorithm
            for i in range(n):
                for j in range(n):
                    # Start from each unvisited black point (value=1)
                    if matrix[i, j] == 1 and visited[i, j] == 0:
                        # Initialize a new region
                        region_size = 0
                        region_coords = []
                        stack = [(i, j)]  # Stack for depth-first search
                        
                        # Depth-first search to find connected black points
                        while stack:
                            x, y = stack.pop()
                            
                            # Skip if out of bounds, already visited, or not black
                            if x < 0 or x >= n or y < 0 or y >= n or visited[x, y] == 1 or matrix[x, y] == 0:
                                continue
                                
                            # Mark as visited and add to region
                            visited[x, y] = 1
                            region_size += 1
                            region_coords.append((x, y))
                            
                            # Add 4-connected neighbors to the stack
                            stack.append((x+1, y))  # Right
                            stack.append((x-1, y))  # Left
                            stack.append((x, y+1))  # Down
                            stack.append((x, y-1))  # Up
                        
                        # Update maximum region if this one is larger
                        if region_size > max_region_size:
                            max_region_size = region_size
                            max_region_coords = region_coords
            
            # Calculate F_a and F_b pattern parameters
            if max_region_coords:
                # Find bounding box of the largest region
                i_min = min(i for i, j in max_region_coords)
                i_max = max(i for i, j in max_region_coords)
                j_min = min(j for i, j in max_region_coords)
                j_max = max(j for i, j in max_region_coords)
                
                # Surface area parameter F_a: size of largest continuous black region
                F_a = max_region_size
                
                # Area ratio parameter F_b: ratio of F_a to the rectangular area enclosing the region
                region_area = (i_max - i_min + 1) * (j_max - j_min + 1)
                F_b = F_a / region_area if region_area > 0 else 0
                
                # Calculate centroid and determine quadrant
                centroid, quadrant = self.calculate_centroid(matrix, max_region_coords)
            else:
                # Default values if no black region found
                F_a = 0
                F_b = 0
                centroid = None
                quadrant = 0
            
            # First check for Type I intermittency
            if abs(F_b - 1.0) < 0.1:  # F_b ≈ 1
                # Type I intermittency detected, check centroid position
                if quadrant == 2:  # Top left quadrant
                    # Confirm as Type I intermittency
                    intermittency_type = "Type I (2nd quadrant)"
                    type_code = 1
                    K_value = None  # No need to compute K for Type I
                elif quadrant == 4:  # Bottom right quadrant
                    # Reclassify as type 2
                    intermittency_type = "Type I (4th quadrant)"
                    type_code = 2
                    K_value = None
                else:
                    # For other quadrants, run the 0-1 test for further classification
                    intermittency_type = f"Type I (quadrant {quadrant})"
                    segment_file = f"U0_{velocity:.2f}_segment_{segment_idx:04d}.npy"
                    segment_path = self.segments_dir / segment_file
                    
                    if segment_path.exists():
                        time_series = np.load(segment_path)
                        K_value = self.zero_one_test.run(time_series)
                        
                        # Apply classification based on K value
                        if K_value <= 0.2:
                            type_code = 0
                            intermittency_type += f", K={K_value:.3f}"
                        elif K_value >= 0.8:
                            type_code = 3
                            intermittency_type += f", K={K_value:.3f}"
                        else:
                            type_code = -1  # Unclassified
                            intermittency_type += f", K={K_value:.3f} (Unclassified)"
                    else:
                        K_value = None
                        type_code = -1  # Unclassified
                        intermittency_type += " (Original signal not found)"
            else:
                # For all non-Type I cases, run the 0-1 test on original time series
                # Find the original time series
                segment_file = f"U0_{velocity:.2f}_segment_{segment_idx:04d}.npy"
                segment_path = self.segments_dir / segment_file
                
                if segment_path.exists():
                    # Load and analyze the original time series with 0-1 test
                    time_series = np.load(segment_path)
                    K_value = self.zero_one_test.run(time_series)
                    
                    # Apply updated classification based on K value
                    if K_value <= 0.2:
                        intermittency_type = "K <= 0.2"
                        type_code = 0
                    elif K_value >= 0.8:
                        intermittency_type = "K >= 0.8"
                        type_code = 3  # Changed from 2 to 3 per request
                    else:
                        intermittency_type = f"Unclassified (K={K_value:.3f})"
                        type_code = -1  # Unclassified - will not be saved
                else:
                    # If original time series not found
                    K_value = None
                    intermittency_type = "Original signal not found"
                    type_code = -1  # Unclassified - will not be saved
            
            # Create and return the metrics dictionary with all analysis results
            metrics = {
                'filename': filename,
                'velocity': velocity,
                'segment_idx': segment_idx,
                'size': n,
                'F_a': F_a,
                'F_b': F_b,
                'centroid': centroid,
                'quadrant': quadrant,
                'K_value': K_value,
                'intermittency_type': intermittency_type,
                'type_code': type_code,
                'matrix': matrix,
                'matrix_path': matrix_path
            }
            
            return metrics
            
        except Exception as e:
            # Handle errors and return basic info with error message
            print(f"Error analyzing matrix {matrix_path}: {e}")
            return {
                'filename': matrix_path.name,
                'error': str(e),
                'intermittency_type': "Error",
                'type_code': -1  # Changed to -1 to be consistent with unclassified
            }
    
    def analyze_all_matrices(self):
        """
        Analyze all recurrence matrices in the folder.
        """
        # Make sure we have the list of matrix files
        if not self.matrix_files:
            self.find_matrix_files()
            
        # Clear previous results
        self.results = []
        total_files = len(self.matrix_files)
        
        # Print starting message
        print(f"Starting analysis of {total_files} matrices...")
        
        # Process each matrix file
        for i, matrix_file in enumerate(self.matrix_files):
            # Print progress updates periodically
            if i % 10 == 0:  # Every 10 files
                print(f"Progress: {i}/{total_files} matrices analyzed ({i/total_files:.1%})")
                
            # Analyze the current matrix and add results to the list
            metrics = self.analyze_matrix(matrix_file)
            self.results.append(metrics)
            
        # Print completion message
        print(f"Analysis completed. Analyzed {len(self.results)} matrices.")
        return self.results
    
    def save_classified_matrices(self):
        """
        Save recurrence matrices with classified filenames.
        Only save matrices with valid type_code (0, 1, 2, 3), skip type_code -1.
        Format: recurrence_matrix_U0_7.00_segment_0000_type_X.npy
        """
        # Check if we have results to save
        if not self.results:
            print("No results to save. Run analyze_all_matrices() first.")
            return
        
        saved_count = 0
        skipped_count = 0
        
        # Iterate through all analysis results
        for result in self.results:
            try:
                # Skip if there was an error in analysis or if type_code is -1
                if 'error' in result or result['type_code'] == -1:
                    skipped_count += 1
                    continue
                
                # Get source file path
                source_path = result['matrix_path']
                
                # Create new filename with type code
                filename_base = os.path.splitext(result['filename'])[0]
                new_filename = f"{filename_base}_type_{result['type_code']}.npy"
                
                # Create destination path
                dest_path = self.output_folder / new_filename
                
                # Copy file with new name
                shutil.copy2(source_path, dest_path)
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving classified matrix {result.get('filename', 'unknown')}: {e}")
        
        print(f"Saved {saved_count} classified matrices to {self.output_folder}")
        print(f"Skipped {skipped_count} unclassified matrices (type_code -1)")
    
    def save_results(self, output_file='intermittency_analysis_results.csv'):
        """
        Save analysis results to a CSV file.
        """
        # Check if we have results to save
        if not self.results:
            print("No results to save. Run analyze_all_matrices() first.")
            return None
            
        try:
            # Create a copy of results without the matrix data for CSV saving
            results_for_csv = []
            for result in self.results:
                result_copy = result.copy()
                if 'matrix' in result_copy:
                    del result_copy['matrix']
                if 'matrix_path' in result_copy:
                    result_copy['matrix_path'] = str(result_copy['matrix_path'])
                # Convert centroid tuple to string if it exists
                if 'centroid' in result_copy and result_copy['centroid'] is not None:
                    result_copy['centroid'] = f"({result_copy['centroid'][0]:.2f}, {result_copy['centroid'][1]:.2f})"
                results_for_csv.append(result_copy)
            
            # Convert to DataFrame for easy CSV output
            df = pd.DataFrame(results_for_csv)
            
            # Create full path for the output file
            output_path = self.output_folder / output_file
            
            # Save the results to CSV
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            
            # Print summary statistics of classifications
            if 'type_code' in df.columns:
                code_counts = df['type_code'].value_counts().sort_index()
                print("\nType Code Distribution:")
                for code, count in code_counts.items():
                    if code == -1:
                        label = "Unclassified"
                    elif code == 0:
                        label = "K <= 0.2"
                    elif code == 1:
                        label = "Type I (2nd quadrant)"
                    elif code == 2:
                        label = "Type I (4th quadrant)"
                    elif code == 3:
                        label = "K >= 0.8"
                    else:
                        label = f"Unknown ({code})"
                    print(f"  {label}: {count} ({count/len(df):.1%})")
            
            return df
        except Exception as e:
            print(f"Error saving results: {e}")
            return None
    
    def visualize_matrices(self, num_matrices=50, max_per_figure=50, cols=5, figsize=(20, 24)):
        """
        Visualize a specified number of recurrence matrices with their type.
        Show only the type number (0, 1, 2, 3) in the title.
        Skip matrices with type_code -1.
        """
        # Check if we have results to visualize
        if not self.results:
            print("No results to visualize. Run analyze_all_matrices() first.")
            return
        
        # Filter out unclassified matrices (type_code -1)
        classified_results = [r for r in self.results if r.get('type_code', -1) != -1 and 'error' not in r]
        
        if not classified_results:
            print("No classified matrices to visualize. All matrices are unclassified.")
            return
        
        # Limit to available matrices
        num_matrices = min(num_matrices, len(classified_results))
        
        # Calculate needed figures (may need multiple figures for large matrix counts)
        num_figures = (num_matrices + max_per_figure - 1) // max_per_figure
        
        # Create figures with grid layouts
        for fig_num in range(num_figures):
            # Calculate start and end indices for this figure
            start_idx = fig_num * max_per_figure
            end_idx = min(start_idx + max_per_figure, num_matrices)
            matrices_in_fig = end_idx - start_idx
            
            # Calculate rows needed for this figure
            rows = (matrices_in_fig + cols - 1) // cols
            
            # Create figure and grid
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(rows, cols)
            
            # Plot each matrix in the grid
            for i, idx in enumerate(range(start_idx, end_idx)):
                result = classified_results[idx]
                
                # Skip if matrix is missing
                if 'matrix' not in result:
                    continue
                
                # Create subplot in the grid
                ax = plt.subplot(gs[i // cols, i % cols])
                
                # Plot the recurrence matrix as a binary image
                ax.imshow(result['matrix'], cmap='binary', interpolation='none')
                
                # Set title with just the type number and appropriate description
                type_code = result['type_code']
                if type_code == 0:
                    title = "Type 0 (K <= 0.2)"
                elif type_code == 1:
                    title = "Type 1 (Type I, 2nd quad)"
                elif type_code == 2:
                    title = "Type 2 (Type I, 4th quad)"
                elif type_code == 3:
                    title = "Type 3 (K >= 0.8)"
                else:
                    title = f"Type {type_code}"
                    
                ax.set_title(title, fontsize=10)
                
                # If centroid exists, plot it on the matrix
                if 'centroid' in result and result['centroid'] is not None:
                    centroid_i, centroid_j = result['centroid']
                    ax.plot(centroid_j, centroid_i, 'r+', markersize=8)
                
                # Remove axis ticks for cleaner visualization
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Create output path and save figure
            output_path = self.output_folder / f"classified_matrices_visualization_{fig_num+1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
            
            # Close the figure to free memory
            plt.close(fig)


# Main execution
if __name__ == "__main__":
    # Create analyzer with automatic path detection
    analyzer = RecurrenceMatricesAnalyzer()
    
    # Find all recurrence matrices
    analyzer.find_matrix_files()
    
    # Only continue if files were found
    if len(analyzer.matrix_files) > 0:
        # Analyze all matrices to extract F_a, F_b and classify types
        analyzer.analyze_all_matrices()
        
        # Save classified matrices with type code in filename
        analyzer.save_classified_matrices()
        
        # Save results to CSV
        analyzer.save_results()
        
        # Visualize the matrices with their classifications
        analyzer.visualize_matrices(num_matrices=450)
        
        print("Analysis, classification, and visualization complete.")
    else:
        print("No matrix files were found. Please check if 'recurrence_matrices' folder exists.")