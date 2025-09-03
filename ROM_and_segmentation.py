"""
Thermoacoustic Segmentation System

This code implements a simplified version of the stability analysis framework that:
1. Simulates thermoacoustic combustion dynamics 
2. Segments pressure signals
3. Saves the segments to a folder without stability classification
4. Generates reversed versions of each segment with continuous naming

The system provides tools for:
- Generating pressure signals for different flow conditions
- Segmenting the signals with sliding windows
- Saving the segments for future analysis
- Creating reversed versions of segments
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import os
import sys

class SegmentHandler:
    """
    Manages the storage of pressure signal segments.
    
    Handles:
    1. Segment storage organization
    2. File naming and directory management
    """
    def __init__(self, base_path=None):
        """
        Initialize segment handler with storage directory.
        
        Args:
            base_path (str): Base directory for segment storage
        """
        if base_path is None:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.join(script_dir, "pressure_segments")
        
        self.base_path = base_path
        
        # Create the base directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
        
        print(f"Segments will be saved to: {self.base_path}")

    def save_segment(self, segment: np.ndarray, U0: float, segment_idx: int):
        """
        Save signal segment with metadata in standardized format.
        
        File naming convention:
        U0_[flow velocity]_segment_[index].npy
        
        Args:
            segment (np.ndarray): Signal segment to save
            U0 (float): Flow velocity parameter
            segment_idx (int): Segment index in sequence
        """
        filename = f"U0_{U0:.2f}_segment_{segment_idx:04d}.npy"
        filepath = os.path.join(self.base_path, filename)
        np.save(filepath, segment)

class CombustionModel:
    """
    Implements a thermoacoustic combustion model with vortex dynamics.
    
    Features:
    - Modal decomposition of acoustic field
    - Vortex formation and tracking
    - Heat release coupling
    - Acoustic-flow interactions
    """
    
    def __init__(self):
        """
        Initialize combustion model parameters and state variables.
        
        Parameters:
        1. Thermodynamic
           - Specific heat ratio (gamma)
           - Speed of sound (c0)
           - Reference density (rho0)
        
        2. Geometric
           - Combustor length (L)
           - Flame location (Lc)
           - Step height (d)
        
        3. Flow Dynamics
           - Mean flow velocity (U0)
           - Strouhal number (St)
           - Convection parameters (alpha0, sigma_alpha)
        """
        # Physical parameters
        self.gamma = 1.4        # Specific heat ratio
        self.c0 = 700.0        # Speed of sound [m/s]
        self.L = 0.7           # Combustor length [m]
        self.Lc = 0.1         # Flame location [m]
        self.d = 0.025         # Step height [m]
        self.xi1 = 29.0        # Base damping rate [1/s]
        self.St = 0.35         # Strouhal number
        self.N = 10            # Number of acoustic modes
        self.beta = 6e03       # Heat release coefficient
        self.rho0 = 1.225      # Reference density [kg/m³]
        
        # Derived parameters
        self.p0 = self.rho0 * self.c0**2 / self.gamma  # Reference pressure
        self.U0 = 8.0          # Mean flow velocity [m/s]
        self.alpha0 = 0.2      # Mean convection ratio
        self.sigma_alpha = 0.02  # Convection variation
        
        # Heat release coupling
        self.c = -2 * (self.gamma - 1) * self.beta / (self.L * self.p0)
        
        # Modal basis
        self.k = np.array([(2 * n - 1) * np.pi / (2 * self.L) for n in range(1, self.N + 1)])
        self.omega = self.c0 * self.k  # Modal frequencies [rad/s]
        
        # State variables
        self.g = np.zeros(self.N)      # Modal amplitudes
        self.g[0] = 0.001              # Initial perturbation
        self.g_dot = np.zeros(self.N)  # Modal velocities
        self.vortices = []             # Active vortices
        self.circulation = 0.0         # Accumulated circulation

    def calculate_damping(self, n: int) -> float:
        """
        Calculate mode-dependent damping coefficient.
        
        Implements quadratic scaling with mode number:
        ξₙ = ξ₁(2n-1)²
        
        Args:
            n (int): Mode number (1-based)
            
        Returns:
            float: Damping coefficient [1/s]
        """
        return self.xi1 * (2 * n - 1) ** 2

    def acoustic_basis_functions(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate acoustic basis functions at position x.
        
        Computes:
        - Pressure modes: cos(kₙx)
        - Velocity modes: sin(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Pressure and velocity mode shapes
        """
        cos_terms = np.cos(self.k * x)
        sin_terms = np.sin(self.k * x)
        return cos_terms, sin_terms

    def calculate_pressure(self, x: float) -> float:
        """
        Calculate acoustic pressure at position x.
        
        Uses modal expansion:
        p'(x,t) = -p₀ Σ (ġₙ/ωₙ)cos(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            float: Acoustic pressure [Pa]
        """
        cos_terms, _ = self.acoustic_basis_functions(x)
        return -self.p0 * np.sum(self.g_dot * cos_terms / self.omega)
    
    def calculate_velocity(self, x: float) -> float:
        """
        Calculate acoustic velocity at position x.
        
        Uses modal expansion:
        u'(x,t) = (c₀/γ) Σ gₙ sin(kₙx)
        
        Args:
            x (float): Axial position [m]
            
        Returns:
            float: Acoustic velocity [m/s]
        """
        _, sin_terms = self.acoustic_basis_functions(x)
        return self.c0 / self.gamma * np.sum(self.g * sin_terms)

    def update_vortex_positions(self, dt: float) -> None:
        """
        Update positions of all active vortices.
        
        Includes:
        1. Mean convection
        2. Acoustic velocity effects
        3. Random turbulent fluctuations
        
        Args:
            dt (float): Time step [s]
        """
        for vortex in self.vortices:
            x, C = vortex  # Position and circulation
            
            # Add random turbulent fluctuation
            alpha = self.alpha0 + self.sigma_alpha * np.random.normal(0, 1)
            
            # Update position using total velocity
            dx = (alpha * self.U0 + self.calculate_velocity(x)) * dt
            vortex[0] += dx

    def update_circulation(self, dt: float) -> None:
        """
        Update circulation accumulation and check for vortex formation.
        
        Process:
        1. Accumulate circulation based on separation velocity
        2. Check against Strouhal criterion
        3. Form new vortex when threshold reached
        
        Args:
            dt (float): Time step [s]
        """
        # Total velocity at separation point
        u_sep = self.U0 + self.calculate_velocity(0)
        
        # Accumulate circulation
        self.circulation += 0.5 * u_sep**2 * dt
        
        # Critical circulation from Strouhal criterion
        C_crit = u_sep * self.d / (2 * self.St)
        
        # Check for vortex formation
        if self.circulation >= C_crit:
            self.vortices.append([0.0, self.circulation])
            self.circulation = 0.0

    def calculate_heat_release(self, C: float) -> np.ndarray:
        """
        Calculate heat release impact on modal velocities.
        
        Models heat release as impulsive forcing when
        vortices reach the flame holder.
        
        Args:
            C (float): Vortex circulation [m²/s]
            
        Returns:
            np.ndarray: Modal velocity impulses
        """
        cos_terms = np.cos(self.k * self.Lc)
        return self.c * C * self.omega * cos_terms

    def handle_vortex_impingement(self) -> None:
        """
        Process vortex-flame interactions.
        
        When vortices reach x = Lc:
        1. Calculate heat release impulse
        2. Apply impulse to modal velocities
        3. Remove the vortex
        """
        for i in range(len(self.vortices) - 1, -1, -1):
            if self.vortices[i][0] >= self.Lc:
                # Extract vortex circulation
                C = self.vortices[i][1]
                
                # Apply heat release impulse
                self.g_dot += self.calculate_heat_release(C)
                
                # Remove vortex
                self.vortices.pop(i)

    def update_modal_amplitudes(self, dt: float) -> None:
        """
        Update modal amplitudes using RK4 integration.
        
        Implements 4th order Runge-Kutta method for the system:
        dg/dt = v
        dv/dt = -2ξv - ω²g
        
        Args:
            dt (float): Time step [s]
        """
        for n in range(self.N):
            # Get mode-specific damping
            xi_n = self.calculate_damping(n + 1)
            
            # Current acceleration
            g_ddot = -xi_n * self.g_dot[n] - self.omega[n]**2 * self.g[n]
            
            # RK4 coefficients for position (k) and velocity (l)
            k1 = dt * self.g_dot[n]
            l1 = dt * g_ddot
            
            k2 = dt * (self.g_dot[n] + 0.5 * l1)
            l2 = dt * (-xi_n * (self.g_dot[n] + 0.5 * l1) - 
                      self.omega[n]**2 * (self.g[n] + 0.5 * k1))
            
            k3 = dt * (self.g_dot[n] + 0.5 * l2)
            l3 = dt * (-xi_n * (self.g_dot[n] + 0.5 * l2) - 
                      self.omega[n]**2 * (self.g[n] + 0.5 * k2))
            
            k4 = dt * (self.g_dot[n] + l3)
            l4 = dt * (-xi_n * (self.g_dot[n] + l3) - 
                      self.omega[n]**2 * (self.g[n] + k3))
            
            # Update using weighted average
            self.g[n] += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.g_dot[n] += (l1 + 2 * l2 + 2 * l3 + l4) / 6

    def simulate(self, t_end: float, dt: float):
        """
        Run full time-domain simulation.
        
        Simulates the coupled system including:
        1. Acoustic field evolution
        2. Vortex dynamics
        3. Heat release coupling
        
        Args:
            t_end (float): End time for simulation [s]
            dt (float): Time step [s]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time and pressure arrays
        """
        # Create time array
        t = np.arange(0, t_end, dt)
        
        # Initialize pressure history
        pressure_history = np.zeros_like(t)
        
        # Main time-stepping loop
        for i, _ in enumerate(t):
            # Record pressure at monitoring point
            pressure_history[i] = self.calculate_pressure(0.09)
            
            # Update physical processes
            self.update_circulation(dt)         # Check for new vortices
            self.update_vortex_positions(dt)    # Move existing vortices
            self.handle_vortex_impingement()    # Process flame interactions
            self.update_modal_amplitudes(dt)    # Advance acoustic field
            
        return t, pressure_history

def process_signals():
    """
    Main processing function for pressure signal segmentation.
    
    Workflow:
    1. Generate pressure signals for different flow velocities
    2. Segment signals using sliding windows
    3. Save segments to pressure_segments folder
    4. Generate and save reversed versions of each segment with continuous numbering
    """
    # Analysis parameters
    t_end = 0.2           # Simulation duration [s]
    dt = 5e-5            # Time step [s]
    window_size = 500    # Analysis window size
    overlap = 250        # Window overlap
    U0_range = np.linspace(7.0, 10.0, 30)  # Flow velocities to test
    
    # Get script directory and set up output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "pressure_segments")
    
    # Initialize segment handler
    handler = SegmentHandler(output_path)
    
    print(f"Starting signal processing. Segments will be saved to {output_path}")
    
    # Counter for continuous segment indexing
    segment_counter = 0
    
    # Process each flow velocity for original segments
    for U0 in tqdm(U0_range, desc="Processing U0 values - Original Segments"):
        # Generate pressure signal
        model = CombustionModel()
        model.U0 = U0
        t, pressure = model.simulate(t_end, dt)
        
        # Store segments for this U0 value for later reversal
        segments_for_U0 = []
        
        # Segment signal with sliding windows
        step = window_size - overlap
        num_windows = (len(pressure) - window_size) // step + 1
        
        for i in range(num_windows):
            # Extract segment
            start = i * step
            end = start + window_size
            if end > len(pressure):
                break
                
            # Save original segment
            segment = pressure[start:end]
            handler.save_segment(segment, U0, segment_counter)
            segment_counter += 1
            
            # Store for later reversal
            segments_for_U0.append(segment)
        
        # Immediately process reversed segments for this U0
        for segment in segments_for_U0:
            # Create and save reversed segment
            reversed_segment = segment[::-1]
            handler.save_segment(reversed_segment, U0, segment_counter)
            segment_counter += 1

    print("\nSegmentation Complete!")
    print(f"Segments have been saved in the {output_path} directory")
    print(f"Total segments generated: {segment_counter}")
    print("All segments use continuous numbering: U0_[flow velocity]_segment_[index].npy")

if __name__ == "__main__":
    process_signals()