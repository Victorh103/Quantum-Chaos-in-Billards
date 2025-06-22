"""
Improved Quantum Chaos in Cardioid Billiard

This script implements an improved version focused on reproducing 
Figure 3 from the paper with better numerical precision.

Adapted to analyze multiple epsilon values: 0.001, 0.05, 0.5, 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import svd
from scipy import special
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time
import os

class CardioidBilliard:
    def __init__(self, epsilon=1.0):
        """Initialize cardioid billiard with parameter epsilon
        For epsilon=0, it's a circle
        For epsilon=1, it's a cardioid
        """
        self.epsilon = epsilon
    
    def boundary(self, phi):
        """Parametrize boundary in polar coordinates"""
        return 1 + self.epsilon * np.cos(phi)
    
    def boundary_cartesian(self, phi):
        """Convert from polar to cartesian coordinates"""
        r = self.boundary(phi)
        return r * np.cos(phi), r * np.sin(phi)
    
    def implicit_function(self, x, y):
        """Implicit function F(x,y) = 0 defining the boundary"""
        if self.epsilon == 0:  # Circle
            return x**2 + y**2 - 1
        elif self.epsilon == 1:  # Cardioid
            return (x**2 + y**2 - 1)**2 - 4*x*(x**2 + y**2)
        else:
            # Approximation for intermediate values
            r_squared = x**2 + y**2
            r = np.sqrt(r_squared)
            phi = np.arctan2(y, x)
            return r - self.boundary(phi)
    
    def plot_billiard(self):
        """Plot the billiard shape"""
        phi = np.linspace(0, 2*np.pi, 1000)
        x, y = self.boundary_cartesian(phi)
        
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, 'k-', linewidth=2)
        plt.axis('equal')
        plt.grid(True)
        plt.title(f'Billiard Shape (ε={self.epsilon})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        
        return plt


class CircleBilliardEigenvalues:
    """Class to compute eigenvalues of circle billiard analytically"""
    
    @staticmethod
    def find_eigenvalues(k_max, max_m=100, max_n=100):
        """Find eigenvalues of the circle billiard up to k_max"""
        eigenvalues = []
        
        # Iterate over m and n
        for m in range(max_m + 1):
            # For each m, find zeros of Bessel function
            zeros = special.jn_zeros(m, max_n)
            
            # Add zeros that are less than k_max
            eigenvalues.extend([zero for zero in zeros if zero <= k_max])
        
        return np.sort(eigenvalues)


class BoundaryIntegralMethod:
    """Class to compute eigenvalues using boundary integral method"""
    
    def __init__(self, billiard, n_boundary_points=300):
        """Initialize with given billiard"""
        self.billiard = billiard
        self.n_boundary_points = n_boundary_points
        
        # Precompute boundary points
        self.phi_values = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
        self.boundary_points = np.array([self.billiard.boundary_cartesian(phi) 
                                         for phi in self.phi_values])
        
        # Precompute boundary derivatives and normal vectors for proper boundary integral
        self.compute_boundary_derivatives()
    
    def compute_boundary_derivatives(self):
        """Compute derivatives and normal vectors at boundary points"""
        phi = self.phi_values
        eps = self.billiard.epsilon
        
        # Derivatives of boundary parametrization
        if eps == 1:  # Cardioid
            dr_dphi = -eps * np.sin(phi)
            x_phi = np.cos(phi) * dr_dphi - np.sin(phi) * (1 + eps * np.cos(phi))
            y_phi = np.sin(phi) * dr_dphi + np.cos(phi) * (1 + eps * np.cos(phi))
        else:  # Circle or other
            dr_dphi = -eps * np.sin(phi)
            x_phi = np.cos(phi) * dr_dphi - np.sin(phi) * (1 + eps * np.cos(phi))
            y_phi = np.sin(phi) * dr_dphi + np.cos(phi) * (1 + eps * np.cos(phi))
        
        # Compute arc length elements
        self.ds_dphi = np.sqrt(x_phi**2 + y_phi**2)
        
        # Compute normal vectors
        norm = np.sqrt(x_phi**2 + y_phi**2)
        self.normals = np.array([((-y_phi[i] / norm[i]), (x_phi[i] / norm[i])) for i in range(len(y_phi))])
    
    def green_function(self, x1, y1, x2, y2, k):
        """Green function for the Helmholtz equation"""
        r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return -0.25j * special.hankel1(0, k * r)
    
    def green_function_derivative(self, x1, y1, x2, y2, nx, ny, k):
        """Normal derivative of Green function"""
        dx = x1 - x2
        dy = y1 - y2
        r = np.sqrt(dx**2 + dy**2)
        
        if r < 1e-10:  # Avoid division by zero
            return 0
        
        # Normal derivative of H0(kr) is -k * H1(kr) * (nx*dx + ny*dy)/r
        return 0.25j * k * special.hankel1(1, k * r) * (nx * dx + ny * dy) / r
    
    def boundary_integral_matrix(self, k):
        """Compute the boundary integral matrix for eigenvalue k²"""
        n = self.n_boundary_points
        matrix = np.zeros((n, n), dtype=complex)
        
        # Fill matrix
        for i in range(n):
            x1, y1 = self.boundary_points[i]
            nx, ny = self.normals[i]
            ds = self.ds_dphi[i] * 2 * np.pi / n
            
            for j in range(n):
                if i == j:
                    # Diagonal elements (analytical limit)
                    matrix[i, i] = -0.25j
                else:
                    # Off-diagonal elements
                    x2, y2 = self.boundary_points[j]
                    # Include arc length element in the integration
                    matrix[i, j] = self.green_function_derivative(x1, y1, x2, y2, nx, ny, k) * ds
        
        return matrix
    
    def singular_value_function(self, k):
        """Compute smallest singular value of boundary integral matrix"""
        matrix = self.boundary_integral_matrix(k)
        # Using scipy.linalg.svd as mentioned in the paper
        s = svd(matrix, compute_uv=False)
        return s[-1]  # Smallest singular value
    
    def find_eigenvalues_in_range(self, k_min, k_max, n_search_points=1000, threshold=1e-6):
        """Find eigenvalues in given range using a much smaller threshold"""
        # Create array of k values
        k_values = np.linspace(k_min, k_max, n_search_points)
        
        # Compute singular values for each k
        singular_values = []
        for k in tqdm(k_values, desc=f"Processing: ", leave=False):
            singular_values.append(self.singular_value_function(k))
        
        singular_values = np.array(singular_values)
        
        # Find local minima of singular values
        candidates = []
        for i in range(1, len(k_values) - 1):
            if (singular_values[i] < singular_values[i-1] and 
                singular_values[i] < singular_values[i+1] and
                singular_values[i] < threshold):
                candidates.append(k_values[i])
        
        # Refine eigenvalues using Brent's method with tighter tolerance
        eigenvalues = []
        for k_approx in candidates:
            k_min_local = k_approx - (k_max - k_min) / n_search_points
            k_max_local = k_approx + (k_max - k_min) / n_search_points
            
            try:
                result = brentq(lambda k: self.singular_value_function(k), 
                               k_min_local, k_max_local, 
                               xtol=1e-12, rtol=1e-10,
                               maxiter=100)
                eigenvalues.append(result)
            except:
                # If brentq fails, use the approximate value
                eigenvalues.append(k_approx)
        
        return np.array(eigenvalues)


# Functions for level spacing distribution with improved unfolding
def unfold_spectrum(eigenvalues):
    """Unfold the spectrum to have mean level spacing of 1"""
    if len(eigenvalues) == 0:
        return np.array([])  # Handle empty array case
        
    n = np.arange(1, len(eigenvalues) + 1)
    
    # Fit a smooth function to the cumulative level counting function
    # For simplicity, use a polynomial of degree 6
    coeffs = np.polyfit(eigenvalues, n, 6)
    smooth_n = np.polyval(coeffs, eigenvalues)
    
    return smooth_n

def compute_level_spacing_distribution(eigenvalues, n_bins=100):
    """Compute the level spacing distribution using spectrum unfolding"""
    if len(eigenvalues) < 2:
        # Return empty arrays if not enough eigenvalues
        return np.array([]), np.array([])
        
    # Ensure eigenvalues are sorted
    eigenvalues = np.sort(eigenvalues)
    
    # Unfold the spectrum
    unfolded = unfold_spectrum(eigenvalues)
    
    # Compute spacings
    spacings = np.diff(unfolded)
    
    # Compute histogram
    hist, bin_edges = np.histogram(spacings, bins=n_bins, range=(0, 4), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, hist


def plot_level_spacing_distribution(eigenvalues, title, n_bins=100):
    """Plot the level spacing distribution and compare with theory"""
    if len(eigenvalues) < 2:
        print(f"Not enough eigenvalues to plot distribution for {title}")
        fig = plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, f"Not enough eigenvalues for {title}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        return fig
        
    # Compute distribution
    bin_centers, hist = compute_level_spacing_distribution(eigenvalues, n_bins)
    
    # Plot distribution
    plt.figure(figsize=(6, 5))
    
    # Plot histogram
    plt.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]), alpha=0.7, 
            label=f'{title} ({len(eigenvalues)} eigenvalues)')
    
    # Plot theoretical curves
    s = np.linspace(0, 4, 1000)
    
    # Poisson distribution (integrable systems)
    poisson = np.exp(-s)
    plt.plot(s, poisson, 'r-', label='exp(-x)')
    
    # GOE (chaotic systems)
    goe = (np.pi/2) * s * np.exp(-(np.pi/4) * s**2)
    plt.plot(s, goe, 'g-', label='GOE')
    
    plt.xlabel('s', fontsize=12)
    plt.ylabel('P(s)', fontsize=12)
    
    plt.xlim(0, 4)
    ax = plt.gca()
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt


# Parallel computation functions
def worker_find_eigenvalues(args):
    """Worker function for parallel computation of eigenvalues"""
    method, k_min, k_max, n_search_points, threshold = args
    return method.find_eigenvalues_in_range(k_min, k_max, n_search_points, threshold)


def parallel_find_eigenvalues(method, k_min, k_max, n_intervals, n_search_points=1000, threshold=1e-6):
    """Find eigenvalues in parallel for multiple k ranges"""
    # Setup multiprocessing
    num_processes = min(mp.cpu_count(), n_intervals)
    print(f"Using {num_processes} processes for parallelization")
    
    # Break up the k range into intervals
    interval_size = (k_max - k_min) / n_intervals
    k_ranges = [(k_min + i*interval_size, k_min + (i+1)*interval_size) 
                for i in range(n_intervals)]
    
    # Prepare tasks
    tasks = [(method, k_range[0], k_range[1], n_search_points, threshold) 
             for k_range in k_ranges]
    
    # Run in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_find_eigenvalues, tasks),
            total=len(tasks),
            desc="Computing eigenvalues"
        ))
    
    # Combine and sort results
    all_eigenvalues = np.concatenate(results)
    
    # Remove duplicates and very close eigenvalues (within numerical precision)
    sorted_vals = np.sort(all_eigenvalues)
    diffs = np.diff(sorted_vals)
    mask = np.append(True, diffs > 1e-10)  # Keep values that are different enough
    
    return sorted_vals[mask]


# Main function with improved parameters
def main():
    # Parameters - IMPROVED VALUES
    epsilon_values = [0]  # Multiple epsilon values to analyze
    n_boundary_points = 1000  # Increased from original
    k_min, k_max = 0.05, 1000.0  # More focused range for quicker results
    n_intervals = min(8, mp.cpu_count())  # Number of intervals for parallelization
    n_search_points = 200    # Points per interval
    eigenvalue_threshold = 0.1  # Less strict threshold to find more eigenvalues
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Store all results
    all_eigenvalues = []
    all_titles = []
    
    # --------------------------------
    # Process each epsilon value
    # --------------------------------
    for epsilon in epsilon_values:
        print(f"\n=== Processing ε = {epsilon} ===")
        start_time = time.time()
        
        if epsilon > 1e-8:  # Use circle billiard analytical solution for small epsilon
            print("Computing eigenvalues analytically (circle billiard)...")
            eigenvalues = CircleBilliardEigenvalues.find_eigenvalues(k_max, max_m=50, max_n=50)
            title = f"Circle-like Billiard (ε={epsilon})"
            np.savetxt(f"results_v3/results_v3/circle_like_eigenvalues_eps_{epsilon}.txt", eigenvalues)
        else:
            print("Computing eigenvalues using boundary integral method...")
            # Create cardioid billiard with specified epsilon
            cardioid = CardioidBilliard(epsilon=epsilon)
            
            
            # Set up boundary integral method
            bim = BoundaryIntegralMethod(cardioid, n_boundary_points=n_boundary_points)
            
            # Find eigenvalues using parallel computation with improved parameters
            print(f"Looking in range k = [{k_min}, {k_max}] with threshold {eigenvalue_threshold}")
            
            # Try to load saved eigenvalues if they exist
            try:
                eigenvalues = np.loadtxt(f"results_v3/results_v3/cardioid_eigenvalues_eps_{epsilon}.txt")
                print(f"Loaded {len(eigenvalues)} eigenvalues from file")
            except:
                # Otherwise compute them
                eigenvalues = parallel_find_eigenvalues(
                    bim, k_min, k_max, n_intervals, 
                    n_search_points, eigenvalue_threshold
                )
                
                if len(eigenvalues) > 0:
                    # Save eigenvalues to file
                    np.savetxt(f"results_v3/results_v3/resultscardioid_eigenvalues_eps_{epsilon}.txt", eigenvalues)
                
                # If we didn't find any eigenvalues, try with a higher threshold
                if len(eigenvalues) == 0:
                    print("No eigenvalues found, trying with a higher threshold...")
                    higher_threshold = 0.5  # Much higher threshold
                    eigenvalues = parallel_find_eigenvalues(
                        bim, k_min, k_max, n_intervals, 
                        n_search_points, higher_threshold
                    )
                    
                    if len(eigenvalues) > 0:
                        # Save eigenvalues to file
                        np.savetxt(f"results_v3/results_v3/cardioid_eigenvalues_eps_{epsilon}.txt", eigenvalues)
                        print(f"Found {len(eigenvalues)} eigenvalues with higher threshold")
            
            title = f"Cardioid Billiard (ε={epsilon})"
        
        print(f"Found {len(eigenvalues)} eigenvalues in {time.time() - start_time:.2f} seconds")
        
        # Store results
        all_eigenvalues.append(eigenvalues)
        all_titles.append(title)
        
        # Plot individual level spacing distribution if we have enough eigenvalues
        if len(eigenvalues) > 1:
            individual_plot = plot_level_spacing_distribution(eigenvalues, title)
            individual_plot.savefig(f"results_v3/results_v3/level_spacing_eps_{epsilon}.png", dpi=300)
            plt.close()
    
    # --------------------------------
    # Combined plots for all epsilon values
    # --------------------------------
    # Create 2x2 subplot for comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (eigenvalues, title, epsilon) in enumerate(zip(all_eigenvalues, all_titles, epsilon_values)):
        ax = axes[i]
        
        if len(eigenvalues) > 1:
            # Compute distribution
            bin_centers, hist = compute_level_spacing_distribution(eigenvalues)
            
            if len(bin_centers) > 0:
                # Plot histogram
                ax.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]) if len(bin_centers) > 1 else 0.1, 
                       alpha=0.7, label=f'{len(eigenvalues)} eigenvalues')
        
        # Plot theoretical curves
        s = np.linspace(0, 4, 1000)
        poisson = np.exp(-s)
        goe = (np.pi/2) * s * np.exp(-(np.pi/4) * s**2)
        
        ax.plot(s, poisson, 'r-', label='exp(-x)')
        ax.plot(s, goe, 'g-', label='GOE')
        ax.set_xlabel('s')
        ax.set_ylabel('P(s)')
        ax.set_title(f'ε = {epsilon}')
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig("results_v3/results_v3/all_epsilon_comparison.png", dpi=300)
    plt.close()
    
    # Create the original Figure 3 style comparison (circle vs cardioid)
    # Only if we have sufficient eigenvalues for both
    circle_eigenvalues = all_eigenvalues[0]  # smallest epsilon
    cardioid_eigenvalues = all_eigenvalues[-1]  # largest epsilon
    
    if len(circle_eigenvalues) > 1 and len(cardioid_eigenvalues) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Circle billiard
        bin_centers, hist = compute_level_spacing_distribution(circle_eigenvalues)
        ax1.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]), alpha=0.7, 
                label=f'circle billiard')
        
        s = np.linspace(0, 4, 1000)
        poisson = np.exp(-s)
        goe = (np.pi/2) * s * np.exp(-(np.pi/4) * s**2)
        
        ax1.plot(s, poisson, 'r-', label='exp(-x)')
        ax1.plot(s, goe, 'g-', label='GOE')
        ax1.set_xlabel('s')
        ax1.set_ylabel('P(s)')
        ax1.set_title('(a)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1.0)
        
        # Cardioid billiard
        bin_centers, hist = compute_level_spacing_distribution(cardioid_eigenvalues)
        ax2.bar(bin_centers, hist, width=(bin_centers[1]-bin_centers[0]), alpha=0.7, 
                label=f'cardioid billiard')
        
        ax2.plot(s, poisson, 'r-', label='exp(-x)')
        ax2.plot(s, goe, 'g-', label='GOE')
        ax2.set_xlabel('s')
        ax2.set_ylabel('P(s)')
        ax2.set_title('(b)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig("results_v3/results_v3/figure3_comparison.png", dpi=300)
        plt.close()
        
        print("\nResults saved in the 'results' directory")
        
        # Print summary of findings
        print("\n=== Summary ===")
        for epsilon, eigenvalues, title in zip(epsilon_values, all_eigenvalues, all_titles):
            print(f"ε = {epsilon}: {len(eigenvalues)} eigenvalues - {title}")
        
        print(f"\nTransition from integrable (ε≈0) to chaotic (ε=1) behavior:")
        print(f"- Small ε → Poisson distribution (integrable system)")
        print(f"- Large ε → GOE distribution (chaotic system)")
    else:
        print("\nWarning: Not enough eigenvalues found to generate meaningful statistics.")
        print("Try adjusting parameters: increase threshold, reduce k range, or increase boundary points.")


if __name__ == "__main__":
    main()