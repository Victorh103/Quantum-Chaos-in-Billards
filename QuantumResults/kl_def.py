"""
Quantum Chaos Analysis: Distribution Fitting - 3-Subplot Combined Plot

This script loads pre-computed eigenvalues for different epsilon values,
computes level spacing distributions with proper normalization and unfolding,
and fits them to Poisson and GOE distributions to quantify the transition 
from integrable to chaotic behavior.

Focus: Creates a clean 3-subplot combined plot showing:
1. R² vs epsilon
2. KL divergence vs epsilon  
3. Number of eigenvalues vs epsilon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency, ks_2samp
import os
import glob
from sklearn.metrics import r2_score

class QuantumChaosAnalyzer:
    def __init__(self, results_dir="results_v3"):
        """Initialize analyzer with results directory"""
        self.results_dir = results_dir
        self.epsilon_values = []
        self.eigenvalues_dict = {}
        self.fit_results = {}
        
    def load_eigenvalues(self):
        """Load all eigenvalue files from results directory"""
        print("Loading eigenvalue files...")
        
        # Look for eigenvalue files
        file_patterns = [
            "circle_like_eigenvalues_eps_*.txt",
            "cardioid_eigenvalues_eps_*.txt",
            "circle_eigenvalues.txt",
            "optimized_eigenvalues_eps_*.txt"
        ]
        
        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(os.path.join(self.results_dir, pattern)))
        
        # Special handling for circle_eigenvalues.txt (epsilon=0)
        circle_file = os.path.join(self.results_dir, "circle_eigenvalues.txt")
        if os.path.exists(circle_file) and circle_file not in all_files:
            all_files.append(circle_file)
        
        for file_path in all_files:
            # Extract epsilon value from filename
            filename = os.path.basename(file_path)
            
            if filename == "circle_eigenvalues.txt":
                epsilon = 0.0
            elif "eps_" in filename:
                eps_str = filename.split("eps_")[1].split(".txt")[0]
                try:
                    epsilon = float(eps_str)
                except ValueError:
                    print(f"Could not parse epsilon from filename: {filename}")
                    continue
            else:
                continue
            
            try:
                eigenvalues = np.loadtxt(file_path)
                
                if len(eigenvalues) > 10:  # Only keep if we have enough eigenvalues
                    self.epsilon_values.append(epsilon)
                    self.eigenvalues_dict[epsilon] = eigenvalues
                    print(f"Loaded ε = {epsilon}: {len(eigenvalues)} eigenvalues")
                else:
                    print(f"Skipped ε = {epsilon}: only {len(eigenvalues)} eigenvalues")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.epsilon_values = sorted(self.epsilon_values)
        print(f"Successfully loaded data for {len(self.epsilon_values)} epsilon values")
    
    def unfold_spectrum_improved(self, eigenvalues):
        """Improved spectrum unfolding with monotonicity enforcement"""
        if len(eigenvalues) == 0:
            return np.array([])
        
        eigenvalues = np.sort(eigenvalues)
        n = np.arange(1, len(eigenvalues) + 1)
        
        try:
            degree = min(3, len(eigenvalues) // 10)
            degree = max(2, degree)
            
            coeffs = np.polyfit(eigenvalues, n, degree)
            smooth_n = np.polyval(coeffs, eigenvalues)
            
            # Ensure monotonicity
            for i in range(1, len(smooth_n)):
                if smooth_n[i] <= smooth_n[i-1]:
                    smooth_n[i] = smooth_n[i-1] + 0.001
            
            return smooth_n
            
        except:
            return np.interp(eigenvalues, eigenvalues, n)
    
    def compute_level_spacing_distribution_accurate(self, eigenvalues, n_bins=50):
        """Compute the level spacing distribution with proper normalization"""
        if len(eigenvalues) < 2:
            return np.array([]), np.array([])
        
        eigenvalues = np.sort(eigenvalues)
        unfolded = self.unfold_spectrum_improved(eigenvalues)
        spacings = np.diff(unfolded)
        spacings = spacings[spacings > 0]
        
        if len(spacings) == 0:
            return np.array([]), np.array([])
        
        hist, bin_edges = np.histogram(spacings, bins=n_bins, range=(0, 4), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Ensure normalization
        normalization = np.sum(hist) * bin_width
        if normalization > 0:
            hist = hist / normalization
        
        return bin_centers, hist
    
    def poisson_distribution(self, s):
        """Theoretical Poisson distribution P(s) = exp(-s)"""
        return np.exp(-s)
    
    def goe_distribution(self, s):
        """Theoretical GOE distribution P(s) = (π/2) * s * exp(-(π/4) * s²)"""
        return (np.pi/2) * s * np.exp(-(np.pi/4) * s**2)
    
    def compute_goodness_of_fit(self, empirical_s, empirical_p, theoretical_func):
        """Compute goodness of fit using R² and KL divergence"""
        if len(empirical_s) == 0 or len(empirical_p) == 0:
            return {'r2': 0, 'kl_div_emp_theo': np.inf}
        
        mask = empirical_p > 0
        if np.sum(mask) < 3:
            return {'r2': 0, 'kl_div_emp_theo': np.inf}
        
        s_clean = empirical_s[mask]
        p_clean = empirical_p[mask]
        theoretical_p = theoretical_func(s_clean)
        
        # R² calculation
        ss_res = np.sum((p_clean - theoretical_p) ** 2)
        ss_tot = np.sum((p_clean - np.mean(p_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # KL divergence calculation
        bin_width = s_clean[1] - s_clean[0] if len(s_clean) > 1 else 0.08
        
        p_empirical_prob = p_clean * bin_width
        p_theoretical_prob = theoretical_p * bin_width
        
        p_empirical_norm = p_empirical_prob / np.sum(p_empirical_prob)
        p_theoretical_norm = p_theoretical_prob / np.sum(p_theoretical_prob)
        
        epsilon = 1e-10
        p_empirical_safe = np.maximum(p_empirical_norm, epsilon)
        p_theoretical_safe = np.maximum(p_theoretical_norm, epsilon)
        
        kl_div_emp_theo = np.sum(p_empirical_safe * np.log(p_empirical_safe / p_theoretical_safe))
        
        return {
            'r2': max(0, r2),
            'kl_div_emp_theo': kl_div_emp_theo
        }
    
    def analyze_epsilon_value(self, epsilon):
        """Analyze a single epsilon value"""
        eigenvalues = self.eigenvalues_dict[epsilon]
        s_values, p_values = self.compute_level_spacing_distribution_accurate(eigenvalues)
        
        if len(s_values) == 0:
            return None
        
        poisson_fit = self.compute_goodness_of_fit(s_values, p_values, self.poisson_distribution)
        goe_fit = self.compute_goodness_of_fit(s_values, p_values, self.goe_distribution)
        
        return {
            'epsilon': epsilon,
            'n_eigenvalues': len(eigenvalues),
            'poisson_r2': poisson_fit['r2'],
            'poisson_kl_emp_theo': poisson_fit['kl_div_emp_theo'],
            'goe_r2': goe_fit['r2'],
            'goe_kl_emp_theo': goe_fit['kl_div_emp_theo']
        }
    
    def run_full_analysis(self):
        """Run analysis for all epsilon values"""
        print("\nRunning full analysis...")
        
        self.fit_results = {}
        
        for epsilon in self.epsilon_values:
            print(f"Analyzing ε = {epsilon}...")
            result = self.analyze_epsilon_value(epsilon)
            
            if result is not None:
                self.fit_results[epsilon] = result
                print(f"  Poisson: R²={result['poisson_r2']:.4f}, KL={result['poisson_kl_emp_theo']:.4f}")
                print(f"  GOE:     R²={result['goe_r2']:.4f}, KL={result['goe_kl_emp_theo']:.4f}")
            else:
                print(f"  Failed to analyze ε = {epsilon}")
    
    def plot_combined_analysis(self, figsize=(12, 5)):
        """Create clean 3-subplot combined plot with seaborn style"""
        if not self.fit_results:
            print("No results to plot!")
            return
        
        # Set seaborn style
        plt.style.use('seaborn-v0_8')
        
        # Extract data for plotting
        epsilons = []
        poisson_r2 = []
        goe_r2 = []
        poisson_kl = []
        goe_kl = []
        n_eigenvalues = []
        
        for epsilon in sorted(self.fit_results.keys()):
            result = self.fit_results[epsilon]
            epsilons.append(epsilon)
            poisson_r2.append(result['poisson_r2'])
            goe_r2.append(result['goe_r2'])
            poisson_kl.append(result['poisson_kl_emp_theo'])
            goe_kl.append(result['goe_kl_emp_theo'])
            n_eigenvalues.append(result['n_eigenvalues'])
        
        epsilons = np.array(epsilons)
        poisson_r2 = np.array(poisson_r2)
        goe_r2 = np.array(goe_r2)
        poisson_kl = np.array(poisson_kl)
        goe_kl = np.array(goe_kl)
        n_eigenvalues = np.array(n_eigenvalues)
        
        # Create clean 3-subplot figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: R² values
        axes[0].semilogx(epsilons, poisson_r2, 'ro-', label='Poisson', linewidth=2, markersize=6)
        axes[0].semilogx(epsilons, goe_r2, 'bo-', label='GOE', linewidth=2, markersize=6)
        axes[0].set_xlabel('ε')
        axes[0].set_ylabel('R²', fontsize=12)
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: KL Divergence
        axes[1].loglog(epsilons, poisson_kl, 'ro-', label='Poisson', linewidth=2, markersize=6)
        axes[1].loglog(epsilons, goe_kl, 'bo-', label='GOE', linewidth=2, markersize=6)
        axes[1].set_xlabel('ε')
        axes[1].set_ylabel('KL divergence', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Fix scientific notation formatting
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        axes[1].yaxis.set_major_formatter(formatter)
        
        # Plot 3: Number of eigenvalues
        axes[2].semilogx(epsilons, n_eigenvalues, 'ko-', linewidth=2, markersize=6)
        axes[2].set_xlabel('ε')
        axes[2].set_ylabel('N', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.results_dir, 'quantum_chaos_3subplot_combined.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
        
        plt.show()
        
        return fig, axes
    
    def print_summary(self):
        """Print a concise summary of results"""
        if not self.fit_results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*50)
        print("QUANTUM CHAOS ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"Analyzed {len(self.fit_results)} epsilon values")
        print(f"Range: ε = {min(self.fit_results.keys()):.6f} to {max(self.fit_results.keys()):.6f}")
        
        # Find transition point
        epsilons = sorted(self.fit_results.keys())
        r2_diffs = []
        for epsilon in epsilons:
            result = self.fit_results[epsilon]
            r2_diff = abs(result['poisson_r2'] - result['goe_r2'])
            r2_diffs.append((epsilon, r2_diff))
        
        transition_eps = min(r2_diffs, key=lambda x: x[1])[0]
        print(f"Estimated transition point: ε ≈ {transition_eps:.4f}")
        
        print(f"\nPhase behavior:")
        print(f"• ε < {transition_eps:.3f}: Integrable (Poisson statistics)")
        print(f"• ε > {transition_eps:.3f}: Chaotic (GOE statistics)")


def main():
    """Main analysis function"""
    print("Quantum Chaos Analysis - 3-Subplot Combined Plot")
    print("="*50)
    
    # Create analyzer
    analyzer = QuantumChaosAnalyzer(results_dir="results_v3/defs")
    
    # Load eigenvalue data
    analyzer.load_eigenvalues()
    
    if not analyzer.epsilon_values:
        print("No eigenvalue files found! Make sure to run the eigenvalue computation first.")
        return
    
    # Run analysis
    analyzer.run_full_analysis()
    
    # Create the requested 3-subplot combined plot
    analyzer.plot_combined_analysis()
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()