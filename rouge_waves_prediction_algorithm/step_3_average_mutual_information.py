import numpy as np
import os
from scipy.signal import savgol_filter

class AMICalculator:
    """
    Class to calculate Average Mutual Information (AMI) for estimating the optimal time delay (tau).
    """
    def __init__(self, time_series, max_lag, bins=30, normalize=True, smooth=True):
        self.time_series = time_series
        self.max_lag = max_lag
        self.bins = bins
        self.normalize = normalize
        self.smooth = smooth
        self.epsilon = 1e-10

        if self.normalize:
            self.time_series = (self.time_series - np.min(self.time_series)) / (
                np.max(self.time_series) - np.min(self.time_series)
            )

    def compute_ami(self):
        """Compute the Average Mutual Information (AMI) for the input time series."""
        ami_values = []

        for tau in range(1, self.max_lag + 1):
            ts_original = self.time_series[:-tau]
            ts_delayed = self.time_series[tau:]

            joint_hist, _, _ = np.histogram2d(ts_original, ts_delayed, bins=self.bins)
            joint_probs = joint_hist / joint_hist.sum()
            marginal_probs1 = joint_probs.sum(axis=0)
            marginal_probs2 = joint_probs.sum(axis=1)

            joint_probs += self.epsilon
            marginal_probs1 += self.epsilon
            marginal_probs2 += self.epsilon

            ami = np.nansum(
                joint_probs * np.log(joint_probs / (marginal_probs1[None, :] * marginal_probs2[:, None]))
            )
            ami_values.append(ami)

        ami_values = np.array(ami_values)

        if self.smooth:
            ami_values = savgol_filter(ami_values, window_length=7, polyorder=3)

        optimal_tau = (np.diff(np.sign(np.diff(ami_values))) > 0).argmax() + 1
        
        # Add validation to ensure reasonable tau value
        if optimal_tau < 1 or np.isnan(optimal_tau):
            optimal_tau = 1
        elif optimal_tau > self.max_lag:
            optimal_tau = self.max_lag

        return optimal_tau

class SegmentedAMIAnalyzer:
    """
    A class to process segmented signals and calculate their optimal time delay.
    """
    def __init__(self, max_lag=500, bins=30, normalize=True, smooth=True):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set input folder as signal_segments_embed and output as signal_segments_embed_tau
        self.input_folder = os.path.join(script_dir, "signal_segments_embed")
        self.output_folder = os.path.join(script_dir, "signal_segments_embed_tau")
        
        # AMI parameters
        self.max_lag = max_lag
        self.bins = bins
        self.normalize = normalize
        self.smooth = smooth

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def process_segmented_signals(self):
        """Process segments to append optimal time delay while preserving embedding dimension."""
        print("\nProcessing segments to append time delays...")
        
        # Get list of all npz files in input folder
        segment_files = [f for f in sorted(os.listdir(self.input_folder)) 
                        if f.endswith('.npz')]
        
        if not segment_files:
            print(f"No .npz files found in {self.input_folder}")
            return
        
        for file_name in segment_files:
            try:
                # Load the segment with embedding dimension
                file_path = os.path.join(self.input_folder, file_name)
                data = np.load(file_path)
                signal = data['signal']
                embedding_dim = data['embedding_dim']

                # Calculate optimal time delay
                calculator = AMICalculator(
                    signal, 
                    self.max_lag, 
                    self.bins, 
                    self.normalize, 
                    self.smooth
                )
                optimal_tau = calculator.compute_ami()

                # Create output filename and save data
                output_path = os.path.join(self.output_folder, file_name)
                np.savez(output_path, 
                        signal=signal, 
                        embedding_dim=embedding_dim,
                        optimal_tau=optimal_tau)
                
                print(f"Processed {file_name} -> Time Delay (tau): {optimal_tau}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")

    def print_summary(self):
        """Print summary of processed segments"""
        print("\nProcessed Files Summary:")
        n_files = len([f for f in os.listdir(self.output_folder) if f.endswith('.npz')])
        print(f"Total segments processed: {n_files}")
        print(f"Files saved in: {self.output_folder}")


# Example usage
if __name__ == "__main__":
    analyzer = SegmentedAMIAnalyzer()
    analyzer.process_segmented_signals()
    analyzer.print_summary()