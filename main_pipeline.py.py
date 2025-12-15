"""
This script executes all 6 modules sequentially to perform complete analysis:
1. ROM_and_segmentation.py - Generate pressure signals
2. cao_theorem.py - Calculate embedding dimensions
3. average_mutual_information.py - Calculate time delays
4. recurrence_matrix_generation.py - Generate recurrence matrices
5. classification.py - Classify stability regimes
6. convolutional_neural_network.py - Train CNN classifier

Usage:
    python main_pipeline.py

Optional: Skip specific modules
    python main_pipeline.py --skip 6  (skip CNN training)
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

class PipelineRunner:
    """Manages execution of the complete extreme event analysis pipeline."""
    
    def __init__(self, skip_modules=None):
        """
        Initialize pipeline runner.
        
        Args:
            skip_modules (list): List of module numbers to skip (1-6)
        """
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.skip_modules = skip_modules if skip_modules else []
        
        # Define all modules in execution order
        self.modules = [
            {
                'number': 1,
                'name': 'ROM and Segmentation',
                'file': 'ROM_and_segmentation.py',
                'description': 'Simulating combustion dynamics and segmenting signals',
            },
            {
                'number': 2,
                'name': 'Cao Theorem',
                'file': 'cao_theorem.py',
                'description': 'Calculating optimal embedding dimensions',
            },
            {
                'number': 3,
                'name': 'Average Mutual Information',
                'file': 'average_mutual_information.py',
                'description': 'Calculating optimal time delays',
            },
            {
                'number': 4,
                'name': 'Recurrence Matrix Generation',
                'file': 'recurrence_matrix_generation.py',
                'description': 'Generating recurrence matrices',
            },
            {
                'number': 5,
                'name': 'Classification',
                'file': 'classification.py',
                'description': 'Classifying stability regimes',
            },
            {
                'number': 6,
                'name': 'CNN Training',
                'file': 'convolutional_neural_network.py',
                'description': 'Training neural network classifier',
            }
        ]
        
        self.total_modules = len(self.modules)
        self.start_time = None
        self.module_times = []
    
    def print_header(self):
        """Print pipeline header."""
        print("=" * 70)
        print("  EXTREME EVENTS ANALYSIS PIPELINE")
        print("=" * 70)
        print(f"\nScript Directory: {self.script_dir}")
        print(f"Python Version: {sys.version.split()[0]}")
        print(f"Total Modules: {self.total_modules}")
        if self.skip_modules:
            print(f"Skipping Modules: {', '.join(map(str, self.skip_modules))}")
        print("\n" + "=" * 70 + "\n")
    
    def print_module_header(self, module):
        """Print header for current module."""
        print("\n" + "=" * 70)
        print(f"  MODULE {module['number']}/{self.total_modules}: {module['name']}")
        print("=" * 70)
        print(f"Description: {module['description']}")
        print(f"File: {module['file']}")
        print(f"Estimated Time: {module['estimated_time']}")
        print("-" * 70 + "\n")
    
    def check_module_exists(self, module):
        """Check if module file exists."""
        module_path = self.script_dir / module['file']
        if not module_path.exists():
            print(f"ERROR: Module file not found: {module['file']}")
            print(f"Expected location: {module_path}")
            return False
        return True
    
    def run_module(self, module):
        """
        Execute a single module.
        
        Args:
            module (dict): Module information dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if module should be skipped
        if module['number'] in self.skip_modules:
            print(f"SKIPPING Module {module['number']}: {module['name']}")
            print("-" * 70 + "\n")
            return True
        
        # Print module header
        self.print_module_header(module)
        
        # Check if file exists
        if not self.check_module_exists(module):
            return False
        
        # Record start time
        module_start = time.time()
        
        # Execute the module
        module_path = self.script_dir / module['file']
        
        try:
            print(f"Executing: python {module['file']}\n")
            
            # Run the module as a subprocess
            result = subprocess.run(
                [sys.executable, str(module_path)],
                cwd=str(self.script_dir),
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            # Check return code
            if result.returncode != 0:
                print(f"\nERROR: Module {module['number']} failed with return code {result.returncode}")
                return False
            
            # Record completion time
            module_time = time.time() - module_start
            self.module_times.append({
                'number': module['number'],
                'name': module['name'],
                'time': module_time
            })
            
            print(f"\nModule {module['number']} completed successfully!")
            print(f"Time taken: {module_time/60:.2f} minutes")
            print("-" * 70)
            
            return True
            
        except Exception as e:
            print(f"\nERROR: Exception occurred while running Module {module['number']}")
            print(f"Error message: {str(e)}")
            return False
    
    def print_summary(self, success):
        """Print final summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 70)
        print("  PIPELINE EXECUTION SUMMARY")
        print("=" * 70 + "\n")
        
        if success:
            print("STATUS: ALL MODULES COMPLETED SUCCESSFULLY!\n")
        else:
            print("STATUS: PIPELINE FAILED\n")
        
        print("Module Execution Times:")
        print("-" * 70)
        for module_time in self.module_times:
            print(f"  Module {module_time['number']}: {module_time['name']}")
            print(f"    Time: {module_time['time']/60:.2f} minutes\n")
        
        print("-" * 70)
        print(f"Total Execution Time: {total_time/60:.2f} minutes")
        print(f"                      ({total_time/3600:.2f} hours)")
        print("=" * 70 + "\n")
        
        if success:
            print("OUTPUT LOCATIONS:")
            print("-" * 70)
            print(f"  Pressure Segments:      {self.script_dir / 'pressure_segments'}/")
            print(f"  Data with Embedding:    {self.script_dir / 'data_with_embed'}/")
            print(f"  Data with Embed & Tau:  {self.script_dir / 'data_with_embed_tau'}/")
            print(f"  Recurrence Matrices:    {self.script_dir / 'recurrence_matrices'}/")
            print(f"  Classified Matrices:    {self.script_dir / 'classified_matrices'}/")
            print(f"  Trained CNN Model:      {self.script_dir / 'recurrence_matrix_cnn.pth'}")
            print(f"  Training Curves:        {self.script_dir / 'training_curves.png'}")
            print(f"  Confusion Matrix:       {self.script_dir / 'confusion_matrix.png'}")
            print("=" * 70 + "\n")
    
    def run_pipeline(self):
        """Execute the complete pipeline."""
        self.start_time = time.time()
        self.print_header()
        
        # Execute each module sequentially
        for module in self.modules:
            success = self.run_module(module)
            
            if not success:
                print("\nPIPELINE ABORTED DUE TO MODULE FAILURE")
                self.print_summary(success=False)
                return False
        
        # Print summary
        self.print_summary(success=True)
        return True


def main():
    """Main entry point for the pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run the complete extreme events analysis pipeline'
    )
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5, 6],
        help='Module numbers to skip (1-6)',
        default=[]
    )
    
    args = parser.parse_args()
    
    # Create pipeline runner
    runner = PipelineRunner(skip_modules=args.skip)
    
    # Run the pipeline
    try:
        success = runner.run_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nPIPELINE INTERRUPTED BY USER")
        print("Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()