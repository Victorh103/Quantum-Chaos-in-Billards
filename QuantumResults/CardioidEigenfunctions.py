import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn_zeros, jn, hankel1
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

class CardioidBilliardEigenfunctions:
    """
    Clase optimizada para calcular autofunciones del billar cardioide
    """
    
    def __init__(self, N_boundary=300):
        self.N_boundary = N_boundary
        self.a = 0.5  # Parámetro del cardioide
        self.setup_boundary()
        
    def setup_boundary(self):
        """Configurar puntos de frontera del cardioide con mejor precisión"""
        t = np.linspace(0, 2*np.pi, self.N_boundary, endpoint=False)
        
        # Cardioide: r = a(1 + cos(θ))
        r = self.a * (1 + np.cos(t))
        
        # Centrar el cardioide: desplazar por -a en x para que el centro quede en (0,0)
        x_centered = r * np.cos(t) - self.a
        y_centered = r * np.sin(t)
        
        self.boundary_points = np.column_stack([x_centered, y_centered])
        
        # Calcular normales con mejor precisión
        dr_dt = -self.a * np.sin(t)
        dx_dt = dr_dt * np.cos(t) - r * np.sin(t)
        dy_dt = dr_dt * np.sin(t) + r * np.cos(t)
        
        # Vector tangente y normal
        tangent = np.column_stack([dx_dt, dy_dt])
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        
        # Evitar división por cero
        tangent_norm = np.maximum(tangent_norm, 1e-12)
        tangent = tangent / tangent_norm
        
        # Normal apuntando hacia adentro
        self.boundary_normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        
        # Elemento diferencial de arco
        dt = 2*np.pi / self.N_boundary
        self.ds = tangent_norm.flatten() * dt
        
    def green_function(self, r1, r2, k):
        """Función de Green 2D optimizada"""
        distance = np.linalg.norm(r1 - r2, axis=-1)
        # Mejor regularización para evitar singularidades
        distance = np.maximum(distance, 1e-10)
        return 1j/4 * hankel1(0, k * distance)
    
    def green_derivative_normal(self, r1, r2, normal, k):
        """Derivada normal de la función de Green optimizada"""
        diff = r1 - r2
        distance = np.linalg.norm(diff, axis=-1, keepdims=True)
        distance = np.maximum(distance, 1e-10)
        
        dot_product = np.sum(diff * normal, axis=-1)
        kr = k * distance.flatten()
        
        # Evitar valores de kr muy pequeños que causan inestabilidad
        kr = np.maximum(kr, 1e-8)
        h1 = hankel1(1, kr)
        
        return 1j*k/4 * h1 * dot_product / distance.flatten()
    
    def build_boundary_integral_matrix(self, k):
        """Construir matriz del sistema BEM con mejor estabilidad"""
        N = self.N_boundary
        matrix = np.zeros((N, N), dtype=complex)
        
        for i in range(N):
            r_i = self.boundary_points[i]
            n_i = self.boundary_normals[i]
            
            for j in range(N):
                r_j = self.boundary_points[j]
                
                if i == j:
                    # Término diagonal mejorado
                    matrix[i, j] = 0.5
                else:
                    # Integral regular
                    val = self.green_derivative_normal(
                        r_i.reshape(1, -1), 
                        r_j.reshape(1, -1), 
                        n_i.reshape(1, -1), 
                        k
                    )[0]
                    matrix[i, j] = val * self.ds[j]
        
        return matrix
    
    def calculate_eigenfunction(self, k, interior_points):
        """Calcular autofunción con mejor manejo de errores"""
        try:
            # Construir matriz del sistema
            matrix = self.build_boundary_integral_matrix(k)
            
            # Verificar que la matriz no sea singular
            cond_number = np.linalg.cond(matrix)
            if cond_number > 1e12:
                print(f"Advertencia: Matriz mal condicionada para k={k} (cond={cond_number:.2e})")
            
            # Resolver para encontrar la función en la frontera usando SVD
            U, s, Vh = svd(matrix)
            
            # Encontrar el vector nulo más estable
            min_idx = np.argmin(np.abs(s))
            boundary_values = Vh[min_idx, :].real
            
            # Normalizar de forma más robusta
            max_val = np.max(np.abs(boundary_values))
            if max_val > 1e-12:
                boundary_values = boundary_values / max_val
            else:
                print(f"Advertencia: Valores de frontera muy pequeños para k={k}")
                return np.zeros(len(interior_points))
            
            # Calcular la función en puntos interiores
            N_interior = len(interior_points)
            psi_interior = np.zeros(N_interior, dtype=complex)
            
            for i, r_int in enumerate(interior_points):
                integral = 0.0
                for j, r_bound in enumerate(self.boundary_points):
                    # Función de Green
                    g = self.green_function(r_int.reshape(1, -1), r_bound.reshape(1, -1), k)[0]
                    
                    # Contribución de la integral
                    integral += boundary_values[j] * g * self.ds[j]
                
                psi_interior[i] = integral
            
            return psi_interior.real
            
        except Exception as e:
            print(f"Error en calculate_eigenfunction para k={k}: {e}")
            return np.zeros(len(interior_points))

def create_centered_cardioid_mask(X, Y, a=0.5):
    """
    Crear máscara del cardioide centrado en el origen
    """
    # Ajustar las coordenadas para el cardioide centrado
    X_shifted = X + a  # Deshacer el centrado para calcular la máscara
    R = np.sqrt(X_shifted**2 + Y**2)
    Phi = np.arctan2(Y, X_shifted)
    
    # Cardioide: r = a(1 + cos(θ))
    cardio_r = a * (1 + np.cos(Phi))
    mask = R <= cardio_r
    
    return mask

def create_cardioid_eigenfunctions_with_given_k(k_values, n_values):
    """
    Crear autofunciones del cardioide con autovalores dados - VERSION CENTRADA
    """
    # Crear billar cardioide
    cardioid_billiard = CardioidBilliardEigenfunctions(N_boundary=20)  # Más puntos para mejor precisión
    
    # Grilla para evaluar las funciones - expandida y centrada
    resolution = 300  # Mayor resolución
    x = np.linspace(-1.2, 0.8, resolution)  # Ajustado para cardioide centrado
    y = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Crear máscara del cardioide centrado
    mask = create_centered_cardioid_mask(X, Y)
    
    # Puntos interiores para evaluar la función
    interior_indices = np.where(mask)
    interior_points = np.column_stack([X[interior_indices], Y[interior_indices]])
    
    eigenfunctions = []
    
    for i, k in enumerate(k_values):
        print(f"Calculando autofunción para n = {n_values[i]}, k = {k:.4f}")
        
        # Calcular autofunción
        psi_values = cardioid_billiard.calculate_eigenfunction(k, interior_points)
        
        # Verificar que se obtuvo una solución válida
        if np.max(np.abs(psi_values)) < 1e-10:
            print(f"Advertencia: Autofunción muy pequeña para k={k}")
        
        # Crear grilla completa
        psi_grid = np.zeros_like(X)
        psi_grid[interior_indices] = psi_values
        psi_grid[~mask] = 0
        
        # Normalizar para visualización con mejor manejo
        psi_squared = psi_grid**2
        max_val = np.max(psi_squared)
        if max_val > 1e-12:
            psi_squared = psi_squared / max_val
        
        eigenfunctions.append((X, Y, psi_squared, mask))
    
    return eigenfunctions

def get_centered_cardioid_boundary():
    """Obtener la frontera del cardioide centrado para visualización"""
    phi_boundary = np.linspace(0, 2*np.pi, 1000)
    a = 0.5
    r_boundary = a * (1 + np.cos(phi_boundary))
    
    # Centrar el cardioide
    x_boundary = r_boundary * np.cos(phi_boundary) - a
    y_boundary = r_boundary * np.sin(phi_boundary)
    
    return x_boundary, y_boundary

def create_complete_comparison_with_given_values(autovalores=30):
    """Crear comparación completa con cardioides centrados y fondo blanco"""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), facecolor='white')
    
    # Configuración
    n_values = [100, 1000, 1500, 2000]
    
    # Autovalores del cardioide - AQUÍ PONES TUS VALORES
    cardioid_k_values = autovalores
    
    print("=== Calculando Billares Cuánticos ===")
    
    # Primera fila: Billar circular (analítico)
    print("\n1. Resolviendo billar circular analíticamente...")
    
    # Grilla común para círculo
    resolution = 300
    x = np.linspace(-1.2, 1.2, resolution)
    y = np.linspace(-1.2, 1.2, resolution)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    
    # Números cuánticos optimizados para cada nivel
    quantum_numbers = [
        (12, 0),   # n = 100
        (25, 5),  # n = 1000  
        (25, 40),  # n = 1500
        (100, 3)   # n = 2000
    ]
    
    for idx, (n, (m, k_order)) in enumerate(zip(n_values, quantum_numbers)):
        ax = axes[0, idx]
        
        try:
            zeros = jn_zeros(m, k_order + 1)
            j_mk = zeros[k_order]
            
            psi = np.zeros_like(X)
            mask = R <= 1.0
            psi[mask] = jn(m, j_mk * R[mask]) * np.cos(m * Phi[mask])
            
            psi_squared = psi**2
            if np.max(psi_squared) > 0:
                psi_squared = psi_squared / np.max(psi_squared)
            
            # Configurar fondo blanco fuera de la máscara
            psi_squared[~mask] = np.nan  # Usar NaN para hacer transparente el exterior
            
            # Visualización mejorada con fondo blanco
            levels = np.linspace(0, 1, 50)
            im = ax.contourf(X, Y, psi_squared, levels=levels, cmap='hot', vmin=0, vmax=1)
            
            print(f"  Circular n={n}: (m={m}, k={k_order}), eigenvalue={j_mk:.3f}")
            
        except Exception as e:
            print(f"  Error en función circular {idx}: {e}")
        
        # Frontera circular
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('white')  # Fondo blanco
        
        # Títulos solo en la fila superior
        ax.set_title(f'N = {n}', fontsize=14, color='black', pad=15)
    
    # Segunda fila: Billar cardioide (numérico) - CENTRADO
    print("\n2. Calculando funciones numéricas para cardioide centrado...")
    
    cardioid_functions = create_cardioid_eigenfunctions_with_given_k(cardioid_k_values, n_values)
    
    # Obtener frontera centrada del cardioide
    x_boundary, y_boundary = get_centered_cardioid_boundary()
    
    for idx, (X_c, Y_c, psi_squared, mask) in enumerate(cardioid_functions):
        ax = axes[1, idx]
        
        # Configurar fondo blanco fuera de la máscara
        psi_squared_display = psi_squared.copy()
        psi_squared_display[~mask] = np.nan  # Usar NaN para hacer transparente el exterior
        
        # Visualización
        levels = np.linspace(0, 1, 50)
        im = ax.contourf(X_c, Y_c, psi_squared_display, levels=levels, cmap='hot', vmin=0, vmax=1)
        
        # Frontera del cardioide centrado
        ax.plot(x_boundary, y_boundary, 'black', linewidth=2)
        
        # Límites ajustados para cardioide centrado
        ax.set_xlim(-1.2, 0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('white')  # Fondo blanco
        
        print(f"  Cardioide n={n_values[idx]}: k={cardioid_k_values[idx]:.3f}")
    
    # Etiquetas de las filas en inglés y en negro
    fig.text(0.02, 0.75, 'Regular Billiard', fontsize=16, 
             rotation=90, va='center', color='black')
    fig.text(0.02, 0.25, 'Chaotic Billiard', fontsize=16, 
             rotation=90, va='center', color='black')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.05)
    
    return fig

def calculate_cardioid_with_your_eigenvalues(your_k_values):
    """
    Función optimizada para calcular solo autofunciones del cardioide
    """
    n_values = [100, 1000, 1500, 2000]
    
    print(f"Usando autovalores dados: {your_k_values}")
    
    # Calcular autofunciones
    cardioid_functions = create_cardioid_eigenfunctions_with_given_k(your_k_values, n_values)
    
    # Crear visualización mejorada
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')
    
    # Obtener frontera centrada del cardioide
    x_boundary, y_boundary = get_centered_cardioid_boundary()
    
    for idx, (X_c, Y_c, psi_squared, mask) in enumerate(cardioid_functions):
        ax = axes[idx]
        
        # Configurar fondo blanco fuera de la máscara
        psi_squared_display = psi_squared.copy()
        psi_squared_display[~mask] = np.nan
        
        # Visualización
        levels = np.linspace(0, 1, 50)
        im = ax.contourf(X_c, Y_c, psi_squared_display, levels=levels, cmap='hot', vmin=0, vmax=1)
        
        # Frontera del cardioide centrado
        ax.plot(x_boundary, y_boundary, 'black', linewidth=2)
        
        # Límites ajustados para cardioide centrado
        ax.set_xlim(-1.2, 0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'$N = {n_values[idx]}, k = {your_k_values[idx]:.3f}$', 
                    fontsize=12, color='black')
        ax.set_facecolor('white')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("=== Autofunciones del Billar Cardioide Corregido ===")
    
    # TUS AUTOVALORES CALCULADOS
    mis_autovalores = [56.895672756, 322.56473765, 479.672330664, 636.487979646]  
    
    
    # Comparación completa (circular + cardioide)
    print("\nCalculando comparación completa...")
    fig2 = create_complete_comparison_with_given_values(mis_autovalores)
    plt.show()
    
    print("\n=== Simulación Completada ===")