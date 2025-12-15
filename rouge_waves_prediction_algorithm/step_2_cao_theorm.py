import numpy as np
from scipy.spatial.distance import cdist
import os

class CaoMethodProcessor:
    def __init__(self, max_dim=20, threshold=0.05):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set input and output folders relative to script location
        self.input_folder = os.path.join(script_dir, "signal_segments")
        self.output_folder = os.path.join(script_dir, "signal_segments_embed")
        self.max_dim = max_dim
        self.threshold = threshold

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def cao_method(self, time_series, tau):
        """
        Compute E1 and E2 statistics using Cao's method with proper error handling.
        """
        time_series = np.array(time_series)
        N = len(time_series)
        E1 = np.zeros(self.max_dim)
        E2 = np.zeros(self.max_dim)

        for d in range(1, self.max_dim):
            if d * tau >= N:
                continue

            max_index = N - (d * tau)
            if max_index <= 0:
                break

            embedding_indices = np.array([np.arange(i, i + d * tau, tau) for i in range(max_index)])
            Y1 = time_series[embedding_indices]
            
            if (d + 1) * tau < N:
                embedding_indices_next = np.array([np.arange(i, i + (d + 1) * tau, tau) 
                                                 for i in range(N - (d + 1) * tau)])
                Y2 = time_series[embedding_indices_next]
            else:
                continue

            N1 = Y1.shape[0]
            N2 = Y2.shape[0]

            if N1 == 0 or N2 == 0:
                continue

            a_d = np.zeros(N1)
            a_d1 = np.zeros(N2)

            for i in range(min(N1, N2)):
                if i >= Y1.shape[0] or i >= Y2.shape[0]:
                    continue

                dist_d = cdist([Y1[i]], Y1).flatten()
                if len(dist_d) > 1:
                    nn_d = np.argsort(dist_d)[1:min(6, len(dist_d))]
                    if len(nn_d) > 0:
                        a_d[i] = np.max(dist_d[nn_d])
                    else:
                        a_d[i] = np.nan

                if i < N2:
                    dist_d1 = cdist([Y2[i]], Y2).flatten()
                    if len(dist_d1) > 1:
                        nn_d1 = np.argsort(dist_d1)[1:min(6, len(dist_d1))]
                        if len(nn_d1) > 0:
                            a_d1[i] = np.max(dist_d1[nn_d1])
                        else:
                            a_d1[i] = np.nan

            valid_indices = ~np.isnan(a_d[:N2]) & ~np.isnan(a_d1[:N2]) & (a_d[:N2] != 0)
            
            if np.any(valid_indices):
                ratio = a_d1[:N2][valid_indices] / a_d[:N2][valid_indices]
                E1[d] = np.mean(ratio[~np.isnan(ratio) & ~np.isinf(ratio)])
            else:
                E1[d] = np.nan

            if (d + 1) * tau < N:
                diffs = np.abs(time_series[((d + 1) * tau):] - time_series[:-((d + 1) * tau)])
                E2[d] = np.mean(diffs) if len(diffs) > 0 else np.nan

        E1 = E1[~np.isnan(E1)]
        E2 = E2[~np.isnan(E2)]

        return E1[1:], E2[1:]

    def calculate_optimal_delay(self, signal):
        """Calculate optimal time delay using autocorrelation."""
        n_points = len(signal)
        max_delay = min(n_points // 3, 100)
        
        signal_normalized = signal - np.mean(signal)
        autocorr = np.correlate(signal_normalized, signal_normalized, mode='full')[n_points-1:]
        autocorr = autocorr / autocorr[0]
        
        for delay in range(1, max_delay):
            if autocorr[delay] <= 0 or (delay > 1 and 
               autocorr[delay] > autocorr[delay-1] and 
               autocorr[delay-1] < autocorr[delay-2]):
                return delay
        return max_delay // 4

    def find_embedding_dimension(self, E1):
        """Determine embedding dimension from E1 values stabilization."""
        if len(E1) < 2:
            return 2
            
        dE1 = np.diff(E1)
        for i, rate in enumerate(dE1):
            if abs(rate) < self.threshold:
                return i + 2
        return len(E1)

    def process_segments(self):
        """Process segments to append embedding dimensions."""
        print("\nProcessing segments to append embedding dimensions...")
        
        # Get list of all npy files in input folder
        segment_files = [f for f in sorted(os.listdir(self.input_folder)) 
                        if f.endswith('.npy')]
        
        if not segment_files:
            print(f"No .npy files found in {self.input_folder}")
            return
        
        for file_name in segment_files:
            try:
                # Load the segment
                file_path = os.path.join(self.input_folder, file_name)
                signal = np.load(file_path)
                
                # Calculate optimal delay and embedding dimension
                optimal_tau = self.calculate_optimal_delay(signal)
                E1, _ = self.cao_method(signal, optimal_tau)
                
                if len(E1) > 0:
                    embedding_dim = self.find_embedding_dimension(E1)
                else:
                    embedding_dim = 2
                
                # Keep the same filename and save in new location
                output_path = os.path.join(self.output_folder, file_name)
                
                # Save both signal and embedding dimension
                np.savez(output_path, signal=signal, embedding_dim=embedding_dim)
                
                print(f"Processed {file_name} -> Embedding Dimension: {embedding_dim}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def print_summary(self):
        """Print summary of processed segments"""
        print("\nProcessed Files Summary:")
        n_files = len([f for f in os.listdir(self.output_folder) if f.endswith('.npy')])
        print(f"Total segments processed: {n_files}")
        print(f"Files saved in: {self.output_folder}")


# Example usage
if __name__ == "__main__":
    processor = CaoMethodProcessor()
    processor.process_segments()
    processor.print_summary()