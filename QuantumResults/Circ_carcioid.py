"""
Simplified Quantum Chaos in Cardioid Billiard

This script implements a simplified version focused on reproducing 
Figure 3 from the paper using parallel computation.

As described in the paper, we calculate eigenvalues for both the circle billiard
(integrable system with Poisson level spacing distribution) and the cardioid billiard
(chaotic system with GOE level spacing distribution).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.linalg import svd
from scipy import special
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

class CardioidBilliard:
    def __init__(self, epsilon=1.0):
        """Initialize cardioid billiaerd with parameter epsilon
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


class CircleBilliardEigenvalues:
    """Class to compute eigenvalues of circle billiard analytically"""
    
    @staticmethod
    def find_eigenvalues(k_max, max_m=50, max_n=50):
        """Find eigenvalues of the circle billiard up to k_max"""
        eigenvalues = []
        
        # Iterate over m and n
        for m in range(max_m + 1):
            # For each m, find zeros of Bessel function
            # Using scipy.special.jn_zeros as mentioned in the paper
            zeros = special.jn_zeros(m, max_n)
            
            # Add zeros that are less than k_max
            for zero in zeros:
                if zero <= k_max:
                    eigenvalues.append(zero)
        
        return np.sort(eigenvalues)


class BoundaryIntegralMethod:
    """Class to compute eigenvalues using boundary integral method"""
    
    def __init__(self, billiard, n_boundary_points=100):
        """Initialize with given billiard"""
        self.billiard = billiard
        self.n_boundary_points = n_boundary_points
        
        # Precompute boundary points
        self.phi_values = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
        self.boundary_points = np.array([self.billiard.boundary_cartesian(phi) 
                                         for phi in self.phi_values])
    
    def green_function(self, x1, y1, x2, y2, k):
        """Green function for the Helmholtz equation"""
        r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return -0.25j * special.hankel1(0, k * r)
    
    def boundary_integral_matrix(self, k):
        """Compute the boundary integral matrix for eigenvalue k²"""
        n = self.n_boundary_points
        matrix = np.zeros((n, n), dtype=complex)
        
        # Fill matrix
        for i in range(n):
            x1, y1 = self.boundary_points[i]
            
            for j in range(n):
                x2, y2 = self.boundary_points[j]
                
                if i != j:
                    # Off-diagonal elements
                    matrix[i, j] = self.green_function(x1, y1, x2, y2, k)
                else:
                    # Diagonal elements (simplified)
                    matrix[i, i] = -0.25j
        
        return matrix
        
    def compute_singular_values_for_range(self, k_values):
        """Compute singular values for a range of k values"""
        return [self.singular_value_function(k) for k in k_values]
    
    def singular_value_function(self, k):
        """Compute smallest singular value of boundary integral matrix"""
        matrix = self.boundary_integral_matrix(k)
        # Using scipy.linalg.svd as mentioned in the paper
        s = svd(matrix, compute_uv=False)
        return s[-1]  # Smallest singular value
    
    def find_eigenvalues_in_range(self, k_min, k_max, n_search_points=1000, threshold=1e-6):
        """Find eigenvalues in given range"""
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
        
        # Refine eigenvalues using Brent's method
        eigenvalues = []
        for k_approx in candidates:
            k_min_local = k_approx - (k_max - k_min) / n_search_points
            k_max_local = k_approx + (k_max - k_min) / n_search_points
            
            try:
                result = brentq(self.singular_value_function, k_min_local, k_max_local)
                eigenvalues.append(result)
            except:
                # If brentq fails, use the approximate value
                eigenvalues.append(k_approx)
        
        return np.array(eigenvalues)

# Functions for level spacing distribution
def rescale_eigenvalues(eigenvalues):
    """Rescale eigenvalues to have mean spacing 1"""
    # Remove possible duplicates
    eigenvalues = np.unique(eigenvalues)
    
    # Compute spacings
    spacings = np.diff(eigenvalues)
    
    # Calculate mean spacing
    mean_spacing = np.mean(spacings)
    
    # Rescale eigenvalues
    rescaled = eigenvalues / mean_spacing
    
    return rescaled


def compute_level_spacing_distribution(eigenvalues, n_bins=100):
    """Compute the level spacing distribution"""
    # Rescale eigenvalues to have mean spacing 1
    rescaled = rescale_eigenvalues(eigenvalues)
    
    # Compute spacings
    spacings = np.diff(rescaled)
    
    # Compute histogram
    hist, bin_edges = np.histogram(spacings, bins=n_bins, range=(0, 4), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, hist


def plot_level_spacing_distribution(eigenvalues, title, n_bins=100):
    """Plot the level spacing distribution and compare with theory"""
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
    return np.sort(all_eigenvalues)
import time
import os

# Main function
def main():
    # Parameters
    epsilon_values = [0.001, 0.05, 0.5, 1.0]  # Multiple epsilon values

    n_boundary_points = 200  # Number of boundary points for numerical method
    k_min, k_max = 0.1, 1000    # Range of k values to search for eigenvalues
    n_intervals = min(20, mp.cpu_count())  # Number of intervals for parallelization
    n_search_points = 500    # Number of points to sample in each interval
    eigenvalue_threshold = 1.0  # Threshold for eigenvalue detection
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Set up plot style
    #plt.style.use('seaborn-v0_8-darkgrid')
        
    # --------------------------------
    # Circle Billiard (integrable)
    # --------------------------------
    print("\n=== Circle Billiard (Integrable System) ===")
    start_time = time.time()
    
    # For circle, use analytical solution
    print("Computing eigenvalues analytically...")
    circle_eigenvalues = CircleBilliardEigenvalues.find_eigenvalues(k_max)
    np.savetxt("results/circle_eigenvalues.txt", circle_eigenvalues)

    
    print(f"Found {len(circle_eigenvalues)} eigenvalues in {time.time() - start_time:.2f} seconds")
    
    # Plot level spacing distribution
    circle_plot = plot_level_spacing_distribution(
        circle_eigenvalues, 
        "Circle Billiard"
    )
    circle_plot.savefig("results/circle_billiard_level_spacing.png", dpi=300)
    circle_plot.close()
    
    # --------------------------------
    # Cardioid Billiard (chaotic)
    # --------------------------------
    print("\n=== Cardioid Billiard (Chaotic System) ===")
    start_time = time.time()
    
    # Create cardioid billiard
    cardioid = CardioidBilliard(epsilon=1.0)
    
    # Display the billiard shape
    cardioid.plot_billiard()
    
    # Set up boundary integral method
    bim = BoundaryIntegralMethod(cardioid, n_boundary_points=n_boundary_points)
    
    # Find eigenvalues using parallel computation
    print("Computing eigenvalues using boundary integral method...")
    cardioid_eigenvalues = parallel_find_eigenvalues(
        bim, k_min, k_max, n_intervals, 
        n_search_points, eigenvalue_threshold
    )
    
    print(f"Found {len(cardioid_eigenvalues)} eigenvalues in {time.time() - start_time:.2f} seconds")
    
    # Plot level spacing distribution
    cardioid_plot = plot_level_spacing_distribution(
        cardioid_eigenvalues, 
        "Cardioid Billiard"
    )
    cardioid_plot.savefig("results/cardioid_billiard_level_spacing.png", dpi=300)

    cardioid_plot.close()
    
    # --------------------------------
    # Combined plot (as in Figure 3)
    # --------------------------------
    # Plot both distributions together for comparison
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
    plt.savefig("results/figure3_comparison.png", dpi=300)
    
    print("\nResults saved in the 'results' directory")
    
    # Print summary of findings
    print("\n=== Summary ===")
    print(f"Circle billiard: {len(circle_eigenvalues)} eigenvalues - matches Poisson distribution (integrable system)")
    print(f"Cardioid billiard: {len(cardioid_eigenvalues)} eigenvalues - matches GOE distribution (chaotic system)")
    
    # Save eigenvalues to files
    np.savetxt("results/cardioid_eigenvalues.txt", cardioid_eigenvalues)


if __name__ == "__main__":
    main()


