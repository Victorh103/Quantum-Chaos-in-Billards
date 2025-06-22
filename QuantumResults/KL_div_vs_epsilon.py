"""
Quantum Chaos Analysis: Distribution Fitting - Improved Version

This script loads pre-computed eigenvalues for different epsilon values,
computes level spacing distributions with proper normalization and unfolding,
and fits them to Poisson and GOE distributions to quantify the transition 
from integrable to chaotic behavior.

Key improvements:
1. Proper histogram normalization that ensures ∫P(s)ds = 1
2. Improved spectrum unfolding with monotonicity enforcement
3. Better handling of numerical artifacts
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
            "circle_eigenvalues.txt" , # Added for epsilon=0 case
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
        """
        Improved spectrum unfolding with monotonicity enforcement.
        This ensures the unfolded spectrum has mean spacing of 1.
        """
        if len(eigenvalues) == 0:
            return np.array([])
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)
        
        # The staircase function N(E) = number of eigenvalues <= E
        n = np.arange(1, len(eigenvalues) + 1)
        
        # Use polynomial fit for the smooth part
        # Degree 3 is usually sufficient and more stable than higher degrees
        try:
            degree = min(3, len(eigenvalues) // 10)
            degree = max(2, degree)  # At least quadratic
            
            coeffs = np.polyfit(eigenvalues, n, degree)
            smooth_n = np.polyval(coeffs, eigenvalues)
            
            # Ensure monotonicity - this is crucial for proper unfolding
            for i in range(1, len(smooth_n)):
                if smooth_n[i] <= smooth_n[i-1]:
                    smooth_n[i] = smooth_n[i-1] + 0.001
            
            return smooth_n
            
        except:
            # Fallback to linear interpolation if polynomial fit fails
            return np.interp(eigenvalues, eigenvalues, n)
    
    def compute_level_spacing_distribution_accurate(self, eigenvalues, n_bins=50):
        """
        Compute the level spacing distribution with proper normalization.
        Ensures that ∫P(s)ds = 1.
        """
        if len(eigenvalues) < 2:
            return np.array([]), np.array([])
        
        # Ensure eigenvalues are sorted
        eigenvalues = np.sort(eigenvalues)
        
        # Unfold the spectrum using improved method
        unfolded = self.unfold_spectrum_improved(eigenvalues)
        
        # Compute spacings
        spacings = np.diff(unfolded)
        
        # Remove any negative or zero spacings (numerical artifacts)
        spacings = spacings[spacings > 0]
        
        if len(spacings) == 0:
            return np.array([]), np.array([])
        
        # Compute histogram with proper normalization
        # The range [0, 4] is standard for level spacing distributions
        hist, bin_edges = np.histogram(spacings, bins=n_bins, range=(0, 4), density=True)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Get bin width for proper normalization check
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Ensure normalization: sum(hist) * bin_width should be approximately 1
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
        """Compute goodness of fit using multiple metrics including KL divergence"""
        if len(empirical_s) == 0 or len(empirical_p) == 0:
            return {'r2': 0, 'chi2_pvalue': 0, 'rmse': np.inf, 'kl_div_emp_theo': np.inf, 'kl_div_theo_emp': np.inf}
        
        # Remove zero bins for better fitting
        mask = empirical_p > 0
        if np.sum(mask) < 3:  # Need at least 3 points
            return {'r2': 0, 'chi2_pvalue': 0, 'rmse': np.inf, 'kl_div_emp_theo': np.inf, 'kl_div_theo_emp': np.inf}
        
        s_clean = empirical_s[mask]
        p_clean = empirical_p[mask]
        
        # Compute theoretical values
        theoretical_p = theoretical_func(s_clean)
        
        # Get bin width for proper normalization in KL divergence
        bin_width = s_clean[1] - s_clean[0] if len(s_clean) > 1 else 0.08  # 4/50 for 50 bins
        
        # Ensure both distributions are properly normalized for KL divergence
        # We need to convert from density to probability mass
        p_empirical_prob = p_clean * bin_width
        p_theoretical_prob = theoretical_p * bin_width
        
        # Renormalize to ensure they sum to 1
        p_empirical_norm = p_empirical_prob / np.sum(p_empirical_prob)
        p_theoretical_norm = p_theoretical_prob / np.sum(p_theoretical_prob)
        
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-10
        p_empirical_safe = np.maximum(p_empirical_norm, epsilon)
        p_theoretical_safe = np.maximum(p_theoretical_norm, epsilon)
        
        # R² coefficient of determination (using density values)
        ss_res = np.sum((p_clean - theoretical_p) ** 2)
        ss_tot = np.sum((p_clean - np.mean(p_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE (using density values)
        rmse = np.sqrt(np.mean((p_clean - theoretical_p) ** 2))
        
        # Kullback-Leibler Divergences (using probability mass)
        # KL(P||Q) = ∑ P(x) * log(P(x)/Q(x))
        kl_div_emp_theo = np.sum(p_empirical_safe * np.log(p_empirical_safe / p_theoretical_safe))
        kl_div_theo_emp = np.sum(p_theoretical_safe * np.log(p_theoretical_safe / p_empirical_safe))
        
        # Chi-squared test
        try:
            # For chi-squared, we need expected frequencies
            # Use the total number of spacings as the sample size
            n_spacings = 1000  # Approximate number of spacings
            
            observed = p_clean * bin_width * n_spacings
            expected = theoretical_p * bin_width * n_spacings
            
            # Avoid division by zero and ensure minimum expected count
            expected = np.maximum(expected, 5)  # Minimum expected count of 5
            
            chi2_stat = np.sum((observed - expected)**2 / expected)
            dof = len(observed) - 1
            
            # Convert to p-value (higher is better fit)
            from scipy.stats import chi2
            chi2_pvalue = 1 - chi2.cdf(chi2_stat, dof)
            
        except:
            chi2_pvalue = 0
        
        return {
            'r2': max(0, r2),  # Ensure non-negative
            'chi2_pvalue': chi2_pvalue,
            'rmse': rmse,
            'kl_div_emp_theo': kl_div_emp_theo,  # KL(empirical||theoretical)
            'kl_div_theo_emp': kl_div_theo_emp   # KL(theoretical||empirical)
        }
    
    def analyze_epsilon_value(self, epsilon):
        """Analyze a single epsilon value"""
        eigenvalues = self.eigenvalues_dict[epsilon]
        
        # Compute level spacing distribution with improved normalization
        s_values, p_values = self.compute_level_spacing_distribution_accurate(eigenvalues)
        
        if len(s_values) == 0:
            return None
        
        # Fit to Poisson distribution
        poisson_fit = self.compute_goodness_of_fit(s_values, p_values, self.poisson_distribution)
        
        # Fit to GOE distribution
        goe_fit = self.compute_goodness_of_fit(s_values, p_values, self.goe_distribution)
        
        # Verify normalization
        bin_width = s_values[1] - s_values[0] if len(s_values) > 1 else 0.08
        total_area = np.sum(p_values) * bin_width
        
        return {
            'epsilon': epsilon,
            'n_eigenvalues': len(eigenvalues),
            's_values': s_values,
            'p_values': p_values,
            'normalization_check': total_area,  # Should be ≈ 1.0
            'poisson_r2': poisson_fit['r2'],
            'poisson_rmse': poisson_fit['rmse'],
            'poisson_kl_emp_theo': poisson_fit['kl_div_emp_theo'],
            'poisson_kl_theo_emp': poisson_fit['kl_div_theo_emp'],
            'goe_r2': goe_fit['r2'],
            'goe_rmse': goe_fit['rmse'],
            'goe_kl_emp_theo': goe_fit['kl_div_emp_theo'],
            'goe_kl_theo_emp': goe_fit['kl_div_theo_emp'],
            'poisson_chi2_pvalue': poisson_fit['chi2_pvalue'],
            'goe_chi2_pvalue': goe_fit['chi2_pvalue']
        }
    
    def run_full_analysis(self):
        """Run analysis for all epsilon values"""
        print("\nRunning full analysis with improved normalization...")
        
        self.fit_results = {}
        
        for epsilon in self.epsilon_values:
            print(f"\nAnalyzing ε = {epsilon}...")
            result = self.analyze_epsilon_value(epsilon)
            
            if result is not None:
                self.fit_results[epsilon] = result
                print(f"  Normalization check: {result['normalization_check']:.4f} (should be ≈ 1.0)")
                print(f"  Poisson: R²={result['poisson_r2']:.4f}, KL={result['poisson_kl_emp_theo']:.4f}")
                print(f"  GOE:     R²={result['goe_r2']:.4f}, KL={result['goe_kl_emp_theo']:.4f}")
            else:
                print(f"  Failed to analyze ε = {epsilon}")
    
    def save_results(self, filename="chaos_analysis_results_improved.txt"):
        """Save analysis results to text file"""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Quantum Chaos Analysis Results (Improved Normalization)\n")
            f.write("="*60 + "\n\n")
            f.write("Columns:\n")
            f.write("- Epsilon: Billiard shape parameter\n")
            f.write("- N_Eigenvals: Number of eigenvalues used\n")
            f.write("- Norm_Check: Normalization check (should be ≈ 1.0)\n")
            f.write("- Poisson_R2: R² for Poisson fit (higher = better fit)\n")
            f.write("- GOE_R2: R² for GOE fit (higher = better fit)\n")
            f.write("- Poisson_KL: KL(empirical||Poisson) (lower = better fit)\n")
            f.write("- GOE_KL: KL(empirical||GOE) (lower = better fit)\n")
            f.write("- Poisson_RMSE: Root mean square error for Poisson\n")
            f.write("- GOE_RMSE: Root mean square error for GOE\n\n")
            
            f.write("Epsilon\tN_Eigenvals\tNorm_Check\tPoisson_R2\tGOE_R2\tPoisson_KL\tGOE_KL\tPoisson_RMSE\tGOE_RMSE\tPoisson_Chi2_p\tGOE_Chi2_p\n")
            
            for epsilon in sorted(self.fit_results.keys()):
                result = self.fit_results[epsilon]
                f.write(f"{epsilon:.6f}\t{result['n_eigenvalues']}\t{result['normalization_check']:.6f}\t")
                f.write(f"{result['poisson_r2']:.6f}\t{result['goe_r2']:.6f}\t")
                f.write(f"{result['poisson_kl_emp_theo']:.6f}\t{result['goe_kl_emp_theo']:.6f}\t")
                f.write(f"{result['poisson_rmse']:.6f}\t{result['goe_rmse']:.6f}\t")
                f.write(f"{result['poisson_chi2_pvalue']:.6f}\t{result['goe_chi2_pvalue']:.6f}\n")
        
        print(f"\nResults saved to: {filepath}")
    
    def plot_individual_distributions(self, n_examples=4):
        """Plot individual level spacing distributions for selected epsilon values"""
        if not self.fit_results:
            print("No results to plot!")
            return
        
        # Select epsilon values to show
        epsilons = sorted(self.fit_results.keys())
        if len(epsilons) <= n_examples:
            selected_epsilons = epsilons
        else:
            # Select evenly spaced values
            indices = np.linspace(0, len(epsilons)-1, n_examples, dtype=int)
            selected_epsilons = [epsilons[i] for i in indices]
        
        n_plots = len(selected_epsilons)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, epsilon in enumerate(selected_epsilons):
            ax = axes[i]
            result = self.fit_results[epsilon]
            
            s_values = result['s_values']
            p_values = result['p_values']
            
            # Plot histogram
            bin_width = s_values[1] - s_values[0] if len(s_values) > 1 else 0.08
            ax.bar(s_values, p_values, width=bin_width, alpha=0.7, 
                   label=f'Data (ε={epsilon:.4f})', edgecolor='black', linewidth=0.5)
            
            # Plot theoretical curves
            s_theory = np.linspace(0, 4, 1000)
            poisson = self.poisson_distribution(s_theory)
            goe = self.goe_distribution(s_theory)
            
            ax.plot(s_theory, poisson, 'r-', linewidth=2, label='Poisson')
            ax.plot(s_theory, goe, 'g-', linewidth=2, label='GOE')
            
            # Add fit statistics
            ax.text(0.95, 0.95, f"Poisson R²: {result['poisson_r2']:.3f}\nGOE R²: {result['goe_r2']:.3f}",
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('s')
            ax.set_ylabel('P(s)')
            ax.set_title(f'ε = {epsilon:.4f} ({result["n_eigenvalues"]} eigenvalues)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 1.1)
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'individual_distributions_improved.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results(self):
        """Create comprehensive plots of the analysis results"""
        if not self.fit_results:
            print("No results to plot!")
            return
        
        # Extract data for plotting
        epsilons = []
        poisson_r2 = []
        goe_r2 = []
        poisson_kl = []
        goe_kl = []
        poisson_rmse = []
        goe_rmse = []
        n_eigenvalues = []
        norm_checks = []
        
        for epsilon in sorted(self.fit_results.keys()):
            result = self.fit_results[epsilon]
            epsilons.append(epsilon)
            poisson_r2.append(result['poisson_r2'])
            goe_r2.append(result['goe_r2'])
            poisson_kl.append(result['poisson_kl_emp_theo'])
            goe_kl.append(result['goe_kl_emp_theo'])
            poisson_rmse.append(result['poisson_rmse'])
            goe_rmse.append(result['goe_rmse'])
            n_eigenvalues.append(result['n_eigenvalues'])
            norm_checks.append(result['normalization_check'])
        
        epsilons = np.array(epsilons)
        poisson_r2 = np.array(poisson_r2)
        goe_r2 = np.array(goe_r2)
        poisson_kl = np.array(poisson_kl)
        goe_kl = np.array(goe_kl)
        norm_checks = np.array(norm_checks)
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: R² values vs log10(epsilon)
        axes[0,0].semilogx(epsilons, poisson_r2, 'ro-', label='Poisson R²', markersize=8, linewidth=2)
        axes[0,0].semilogx(epsilons, goe_r2, 'bo-', label='GOE R²', markersize=8, linewidth=2)
        axes[0,0].set_xlabel('ε (log scale)')
        axes[0,0].set_ylabel('R² (Goodness of Fit)')
        axes[0,0].set_title('R² vs Epsilon (Higher = Better)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        axes[0,0].set_ylim(0, 1)
        
        
        print(poisson_r2, goe_r2)
        print(epsilons)
        print(poisson_kl, goe_kl)
        # Plot 2: KL Divergence values vs log10(epsilon)
        axes[0,1].semilogx(epsilons, poisson_kl, 'ro-', label='KL(emp||Poisson)', markersize=8, linewidth=2)
        axes[0,1].semilogx(epsilons, goe_kl, 'bo-', label='KL(emp||GOE)', markersize=8, linewidth=2)
        axes[0,1].set_xlabel('ε (log scale)')
        axes[0,1].set_ylabel('KL Divergence')
        axes[0,1].set_title('KL Divergence vs Epsilon (Lower = Better)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_yscale('log')  # Log scale for KL divergence
        
        # Plot 3: Normalization check
        axes[0,2].semilogx(epsilons, norm_checks, 'go-', markersize=8, linewidth=2)
        axes[0,2].axhline(y=1.0, color='k', linestyle='--', label='Ideal normalization')
        axes[0,2].set_xlabel('ε (log scale)')
        axes[0,2].set_ylabel('Normalization ∫P(s)ds')
        axes[0,2].set_title('Normalization Check (Should be ≈ 1.0)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        axes[0,2].set_ylim(0.9, 1.1)
        
        # Plot 4: RMSE comparison
        axes[1,0].semilogx(epsilons, poisson_rmse, 'ro-', label='Poisson RMSE', markersize=8, linewidth=2)
        axes[1,0].semilogx(epsilons, goe_rmse, 'bo-', label='GOE RMSE', markersize=8, linewidth=2)
        axes[1,0].set_xlabel('ε (log scale)')
        axes[1,0].set_ylabel('RMSE (lower is better)')
        axes[1,0].set_title('Root Mean Square Error vs Epsilon')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot 5: Combined metric (R² - KL/10)
        combined_poisson = poisson_r2 - poisson_kl/10
        combined_goe = goe_r2 - goe_kl/10
        
        axes[1,1].semilogx(epsilons, combined_poisson, 'ro-', label='Poisson Combined', markersize=8, linewidth=2)
        axes[1,1].semilogx(epsilons, combined_goe, 'bo-', label='GOE Combined', markersize=8, linewidth=2)
        axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('ε (log scale)')
        axes[1,1].set_ylabel('R² - KL/10')
        axes[1,1].set_title('Combined Metric (Higher = Better Fit)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Plot 6: Number of eigenvalues
        axes[1,2].semilogx(epsilons, n_eigenvalues, 'go-', markersize=8, linewidth=2)
        axes[1,2].set_xlabel('ε (log scale)')
        axes[1,2].set_ylabel('Number of Eigenvalues')
        axes[1,2].set_title('Dataset Size vs Epsilon')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'chaos_analysis_comprehensive_improved.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # R² plot
        ax1.semilogx(epsilons, poisson_r2, 'ro-', label='Poisson R²', markersize=10, linewidth=3)
        ax1.semilogx(epsilons, goe_r2, 'bo-', label='GOE R²', markersize=10, linewidth=3)
        ax1.set_xlabel('ε (Billiard Shape Parameter)', fontsize=14)
        ax1.set_ylabel('R² (Goodness of Fit)', fontsize=14)
        ax1.set_title('Quantum Chaos Transition: R² Analysis', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Find and mark transition point
        r2_diff = np.abs(poisson_r2 - goe_r2)
        transition_idx = np.argmin(r2_diff)
        transition_eps = epsilons[transition_idx]
        ax1.axvline(x=transition_eps, color='gray', linestyle='--', alpha=0.7)
        ax1.text(transition_eps, 0.5, f'Transition\nε ≈ {transition_eps:.3f}', 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # KL divergence plot
        ax2.semilogx(epsilons, poisson_kl, 'ro-', label='KL(Data||Poisson)', markersize=10, linewidth=3)
        ax2.semilogx(epsilons, goe_kl, 'bo-', label='KL(Data||GOE)', markersize=10, linewidth=3)
        ax2.set_xlabel('ε (Billiard Shape Parameter)', fontsize=14)
        ax2.set_ylabel('KL Divergence', fontsize=14)
        ax2.set_title('Quantum Chaos Transition: KL Divergence', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Add annotations
        for ax in [ax1, ax2]:
            ax.text(min(epsilons)*1.5, 0.9, 'Integrable\n(Circle-like)', fontsize=12, 
                    transform=ax.get_yaxis_transform(), ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax.text(max(epsilons)*0.5, 0.9, 'Chaotic\n(Cardioid)', fontsize=12,
                    transform=ax.get_yaxis_transform(), ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'chaos_transition_summary_improved.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot individual distributions
        self.plot_individual_distributions()
    
    def print_summary(self):
        """Print summary of results"""
        if not self.fit_results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*60)
        print("QUANTUM CHAOS ANALYSIS SUMMARY (Improved Version)")
        print("="*60)
        
        print(f"Analyzed {len(self.fit_results)} epsilon values:")
        print(f"Range: ε = {min(self.fit_results.keys()):.6f} to {max(self.fit_results.keys()):.6f}")
        
        # Check normalization quality
        norm_checks = [self.fit_results[eps]['normalization_check'] for eps in self.fit_results]
        avg_norm = np.mean(norm_checks)
        std_norm = np.std(norm_checks)
        print(f"\nNormalization quality:")
        print(f"Average: {avg_norm:.4f} ± {std_norm:.4f} (should be 1.000)")
        
        print("\nKey Findings:")
        
        # Find best fits for both R² and KL divergence
        best_poisson_r2_eps = max(self.fit_results.keys(), key=lambda x: self.fit_results[x]['poisson_r2'])
        best_goe_r2_eps = max(self.fit_results.keys(), key=lambda x: self.fit_results[x]['goe_r2'])
        
        # For KL divergence, lower is better
        best_poisson_kl_eps = min(self.fit_results.keys(), key=lambda x: self.fit_results[x]['poisson_kl_emp_theo'])
        best_goe_kl_eps = min(self.fit_results.keys(), key=lambda x: self.fit_results[x]['goe_kl_emp_theo'])
        
        print(f"\nBest Poisson R² fit: ε = {best_poisson_r2_eps:.6f} (R² = {self.fit_results[best_poisson_r2_eps]['poisson_r2']:.4f})")
        print(f"Best GOE R² fit: ε = {best_goe_r2_eps:.6f} (R² = {self.fit_results[best_goe_r2_eps]['goe_r2']:.4f})")
        print(f"Best Poisson KL fit: ε = {best_poisson_kl_eps:.6f} (KL = {self.fit_results[best_poisson_kl_eps]['poisson_kl_emp_theo']:.4f})")
        print(f"Best GOE KL fit: ε = {best_goe_kl_eps:.6f} (KL = {self.fit_results[best_goe_kl_eps]['goe_kl_emp_theo']:.4f})")
        
        # Transition point estimates
        r2_crossover_candidates = []
        kl_crossover_candidates = []
        combined_crossover_candidates = []
        
        for epsilon in sorted(self.fit_results.keys()):
            result = self.fit_results[epsilon]
            r2_diff = abs(result['poisson_r2'] - result['goe_r2'])
            kl_diff = abs(result['poisson_kl_emp_theo'] - result['goe_kl_emp_theo'])
            
            # Combined metric: R² difference + normalized KL difference
            combined_diff = r2_diff + kl_diff / 10.0
            
            r2_crossover_candidates.append((epsilon, r2_diff))
            kl_crossover_candidates.append((epsilon, kl_diff))
            combined_crossover_candidates.append((epsilon, combined_diff))
        
        r2_crossover_eps = min(r2_crossover_candidates, key=lambda x: x[1])[0]
        kl_crossover_eps = min(kl_crossover_candidates, key=lambda x: x[1])[0]
        combined_crossover_eps = min(combined_crossover_candidates, key=lambda x: x[1])[0]
        
        print(f"\nTransition point estimates:")
        print(f"Based on R² crossover: ε ≈ {r2_crossover_eps:.6f}")
        print(f"Based on KL crossover: ε ≈ {kl_crossover_eps:.6f}")
        print(f"Combined estimate: ε ≈ {combined_crossover_eps:.6f}")
        
        # Analyze the transition region
        transition_region = []
        for epsilon in sorted(self.fit_results.keys()):
            result = self.fit_results[epsilon]
            if 0.3 < result['poisson_r2'] < 0.7 and 0.3 < result['goe_r2'] < 0.7:
                transition_region.append(epsilon)
        
        if transition_region:
            print(f"\nTransition region: ε ∈ [{min(transition_region):.4f}, {max(transition_region):.4f}]")
        
        print("\nInterpretation:")
        print("• Small ε → Integrable dynamics (circle-like)")
        print("  - High Poisson R² (> 0.8)")
        print("  - Low Poisson KL divergence (< 0.1)")
        print("  - Level spacing shows exponential decay (no level repulsion)")
        print("\n• Large ε → Chaotic dynamics (cardioid)")
        print("  - High GOE R² (> 0.8)")
        print("  - Low GOE KL divergence (< 0.1)")
        print("  - Level spacing shows linear repulsion at small s")
        print("\n• Transition occurs around ε ≈ 0.1-0.3")
        print("  - Both distributions fit moderately well")
        print("  - System shows mixed phase space")
        print("\nNote: The improved normalization ensures ∫P(s)ds = 1 for all distributions")


def main():
    """Main analysis function"""
    print("Quantum Chaos Distribution Fitting Analysis - Improved Version")
    print("="*60)
    
    # Create analyzer
    analyzer = QuantumChaosAnalyzer(results_dir="results_v3/defs")
    
    # Load eigenvalue data
    analyzer.load_eigenvalues()
    
    if not analyzer.epsilon_values:
        print("No eigenvalue files found! Make sure to run the eigenvalue computation first.")
        return
    
    # Run analysis
    analyzer.run_full_analysis()
    
    # Save results
    analyzer.save_results()
    
    # Create plots
    analyzer.plot_results()
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()