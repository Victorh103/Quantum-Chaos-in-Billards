import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, entropy
import warnings
warnings.filterwarnings('ignore')

def read_eigenvalues_from_file(filename='autovaloresepsilon1.txt'):
    """
    Leer los autovalores exactos del archivo de texto
    """
    try:
        with open(filename, 'r') as file:
            eigenvalues = []
            for line in file:
                eigenvalues.append(float(line.strip()))
        return np.array(eigenvalues)
    except FileNotFoundError:
        print(f"Archivo {filename} no encontrado.")
        return None

def calculate_kl_divergence(observed_k, theoretical_k):
    """
    Calcular la divergencia KL entre los valores observados y teóricos
    """
    # Normalizar para crear distribuciones de probabilidad
    observed_prob = observed_k / np.sum(observed_k)
    theoretical_prob = theoretical_k / np.sum(theoretical_k)
    
    # Evitar valores cero añadiendo un pequeño epsilon
    epsilon = 1e-10
    observed_prob = observed_prob + epsilon
    theoretical_prob = theoretical_prob + epsilon
    
    # Renormalizar
    observed_prob = observed_prob / np.sum(observed_prob)
    theoretical_prob = theoretical_prob / np.sum(theoretical_prob)
    
    # Calcular KL divergence
    kl_div = entropy(observed_prob, theoretical_prob)
    return kl_div

def calculate_residuals_statistics(observed, predicted):
    """
    Calcular estadísticas de los residuos
    """
    residuals = observed - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    
    return {
        'residuals': residuals,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error
    }

def analyze_weyl_law_enhanced():
    """
    Análisis mejorado de la ley de Weyl con KL divergence y estadísticas adicionales
    """
    print("=== Análisis Mejorado de la Ley de Weyl: k vs √n ===")
    
    # Leer autovalores del archivo
    eigenvalues = read_eigenvalues_from_file()
    if eigenvalues is None:
        return None
    
    print(f"Total de autovalores leídos: {len(eigenvalues)}")
    
    # Verificar que los autovalores estén ordenados
    if not np.all(eigenvalues[:-1] <= eigenvalues[1:]):
        print("¡ADVERTENCIA! Los autovalores no están ordenados. Ordenando...")
        eigenvalues = np.sort(eigenvalues)
    
    # Índices específicos
    indices = [100, 300, 500, 700, 1000, 1500, 2000, 2500, 3000]
    
    # Extraer k = √λ para estos índices
    selected_k_values = []
    selected_indices = []
    
    for idx in indices:
        if idx <= len(eigenvalues):
            lambda_val = eigenvalues[idx-1]  # λ_n
            k_val = np.sqrt(lambda_val)      # k = √λ_n
            selected_k_values.append(k_val)
            selected_indices.append(idx)
            print(f"n = {idx:4d}: λ = {lambda_val:.6f}, k = {k_val:.6f}")
    
    selected_k_values = np.array(selected_k_values)
    selected_indices = np.array(selected_indices)
    
    # Ajuste lineal k vs √n
    sqrt_n = np.sqrt(selected_indices)
    slope, intercept, r_value, p_value, std_err = linregress(sqrt_n, selected_k_values)
    
    # Valores teóricos del ajuste
    k_theoretical = slope * sqrt_n + intercept
    
    # Calcular KL divergence
    kl_divergence = calculate_kl_divergence(selected_k_values, k_theoretical)
    
    # Calcular estadísticas de residuos
    residuals_stats = calculate_residuals_statistics(selected_k_values, k_theoretical)
    
    print(f"\n=== Resultados del Análisis ===")
    print(f"Ecuación del ajuste: k = {slope:.6f}√n + {intercept:.6f}")
    print(f"Coeficiente de correlación: R² = {r_value**2:.8f}")
    print(f"Error estándar de la pendiente: {std_err:.8f}")
    print(f"Valor p: {p_value:.2e}")
    print(f"\n=== Análisis de Divergencia ===")
    print(f"KL Divergence: {kl_divergence:.8f}")
    print(f"RMSE: {residuals_stats['rmse']:.8f}")
    print(f"MAE: {residuals_stats['mae']:.8f}")
    print(f"Error máximo: {residuals_stats['max_error']:.8f}")
    
    return {
        'indices': selected_indices,
        'k_values': selected_k_values,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'kl_divergence': kl_divergence,
        'residuals_stats': residuals_stats,
        'k_theoretical': k_theoretical,
        'sqrt_n': sqrt_n
    }

def create_weyl_plot():
    """
    Crear gráfica de k vs √n para informe de física
    """
    results = analyze_weyl_law_enhanced()
    
    if results is None:
        print("No se pudieron cargar los datos.")
        return None
    
    # Configurar estilo para publicación científica
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Gráfica principal
    sqrt_n = results['sqrt_n']
    k_values = results['k_values']
    slope = results['slope']
    intercept = results['intercept']
    r_value = results['r_value']
    kl_div = results['kl_divergence']
    
    # Datos experimentales
    ax.scatter(sqrt_n, k_values, color='#2E86AB', s=80, alpha=0.8, 
               label='Experimental data', edgecolors='black', linewidth=0.5)
    
    # Línea de ajuste
    sqrt_n_fit = np.linspace(min(sqrt_n), max(sqrt_n), 100)
    k_fit = slope * sqrt_n_fit + intercept
    ax.plot(sqrt_n_fit, k_fit, color='#A23B72', linewidth=2.5,
             label=f'Linear fit: k = {slope:.4f}√n + {intercept:.4f}')
    
    # Barras de error (estimadas)
    error_bars = np.full_like(k_values, results['residuals_stats']['rmse'])
    ax.errorbar(sqrt_n, k_values, yerr=error_bars, fmt='none', 
                color='#2E86AB', alpha=0.3, capsize=3)
    
    ax.set_xlabel('√n', fontsize=14)
    ax.set_ylabel('k', fontsize=14)
    ax.set_title('Weils law for Cardioid Billiard (ε = 1)', 
                 fontsize=16, pad=20)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Caja de estadísticas
    stats_text = f'R² = {r_value**2:.6f}\n' + \
                f'KL Divergence = {kl_div:.6f}\n'     
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="white", alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()

    return fig, results

def generate_analysis_report():
    """
    Generar un reporte completo del análisis
    """
    results = analyze_weyl_law_enhanced()
    if results is None:
        return
    
    print("\n" + "="*70)
    print("REPORTE COMPLETO DEL ANÁLISIS DE LA LEY DE WEYL")
    print("="*70)
    
    print(f"\n1. PARÁMETROS DEL AJUSTE LINEAL:")
    print(f"   • Pendiente (m): {results['slope']:.8f} ± {results['std_err']:.8f}")
    print(f"   • Intercepto (b): {results['intercept']:.8f}")
    print(f"   • Ecuación: k = {results['slope']:.6f}√n + {results['intercept']:.6f}")
    
    print(f"\n2. CALIDAD DEL AJUSTE:")
    print(f"   • Coeficiente de determinación (R²): {results['r_value']**2:.8f}")
    print(f"   • Coeficiente de correlación (R): {results['r_value']:.8f}")
    print(f"   • Valor p: {results['p_value']:.2e}")
    
    print(f"\n3. ANÁLISIS DE DIVERGENCIA:")
    print(f"   • KL Divergence: {results['kl_divergence']:.8f}")
    print(f"   • RMSE: {results['residuals_stats']['rmse']:.8f}")
    print(f"   • MAE: {results['residuals_stats']['mae']:.8f}")
    print(f"   • Error máximo: {results['residuals_stats']['max_error']:.8f}")
    
    print(f"\n4. INTERPRETACIÓN FÍSICA:")
    teorica_slope = np.pi / 2  
    deviation = abs(results['slope'] - teorica_slope) / teorica_slope * 100
    print(f"   • Pendiente teórica esperada: {teorica_slope:.6f}")
    print(f"   • Desviación de la teoría: {deviation:.2f}%")
    
    if results['kl_divergence'] < 0.01:
        print("   • Excelente concordancia con la Ley de Weyl")
    elif results['kl_divergence'] < 0.05:
        print("   • Buena concordancia con la Ley de Weyl")
    else:
        print("   • Concordancia moderada con la Ley de Weyl")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("=== Análisis Mejorado de la Ley de Weyl ===")
    
    # Generar reporte completo
    generate_analysis_report()
    
    # Crear gráfica profesional
    fig, results = create_weyl_plot()
    
    if fig is not None:
        plt.show()
        print("\n=== Gráfica profesional completada ===")
        print("La gráfica está lista para incluir en tu informe de física.")
        
        # Opción para guardar
        save_option = input("\n¿Deseas guardar la gráfica? (s/n): ")
        if save_option.lower() == 's':
            filename = input("Nombre del archivo (sin extensión): ")
            if not filename:
                filename = "weyl_law_analysis"
            fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{filename}.pdf", bbox_inches='tight')
            print(f"Gráfica guardada como {filename}.png y {filename}.pdf")
    else:
        print("No se pudo generar la gráfica.")