import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def billiard_boundary(phi, epsilon):
    """Define la frontera del billar parametrizada en coordenadas polares."""
    return 1 + epsilon * np.cos(phi)

def boundary_point(phi, epsilon):
    """Calcula el punto en la frontera para un ángulo dado."""
    r = billiard_boundary(phi, epsilon)
    return r * np.cos(phi), r * np.sin(phi)

def boundary_derivative(phi, epsilon):
    """Calcula la derivada de la frontera respecto a phi."""
    return -epsilon * np.sin(phi)

def normal_vector(phi, epsilon):
    """Calcula el vector normal unitario en un punto de la frontera."""
    r = billiard_boundary(phi, epsilon)
    dr_dphi = boundary_derivative(phi, epsilon)
    
    # Vector tangente
    tx = -r * np.sin(phi) + dr_dphi * np.cos(phi)
    ty = r * np.cos(phi) + dr_dphi * np.sin(phi)
    
    # Normalización del vector tangente
    norm_t = np.sqrt(tx**2 + ty**2)
    if norm_t > 1e-12:  # Evitar división por cero
        tx /= norm_t
        ty /= norm_t
    
    # Vector normal (perpendicular al tangente)
    nx, ny = -ty, tx
    
    return nx, ny

def reflect_velocity(vx, vy, nx, ny):
    """Refleja el vector velocidad respecto a la normal."""
    v_dot_n = vx * nx + vy * ny
    vx_new = vx - 2 * v_dot_n * nx
    vy_new = vy - 2 * v_dot_n * ny
    return vx_new, vy_new

def implicit_boundary_function(x, y, epsilon):
    """Función implícita F(x,y) = 0 que define la frontera del billar."""
    r = np.sqrt(x**2 + y**2)
    if r < 1e-12:  # Evitar división por cero en el origen
        return -1.0
    cos_phi = x / r
    return r - (1 + epsilon * cos_phi)

def find_next_intersection(x, y, vx, vy, epsilon, max_t=20.0):
    """Encuentra la siguiente intersección de la trayectoria con la frontera."""
    def f(t):
        if t <= 0:
            return float('inf')
        return implicit_boundary_function(x + t * vx, y + t * vy, epsilon)
    
    # Buscar intervalo con cambio de signo
    t_min = 1e-8
    t_max = max_t
    n_search = 2000
    
    t_values = np.linspace(t_min, t_max, n_search)
    f_values = np.array([f(t) for t in t_values])
    
    # Encontrar cambios de signo
    sign_changes = np.where(np.diff(np.signbit(f_values)))[0]
    
    if len(sign_changes) == 0:
        raise ValueError("No se encontró intersección con la frontera")
    
    # Usar el primer cambio de signo encontrado
    idx = sign_changes[0]
    t_left = t_values[idx]
    t_right = t_values[idx + 1]
    
    # Refinar con brentq
    try:
        t = brentq(f, t_left, t_right, xtol=1e-10, maxiter=100)
    except (ValueError, RuntimeError):
        # Si falla, usar bisección manual
        for _ in range(50):
            t_mid = 0.5 * (t_left + t_right)
            if f(t_left) * f(t_mid) < 0:
                t_right = t_mid
            else:
                t_left = t_mid
            if abs(t_right - t_left) < 1e-10:
                break
        t = 0.5 * (t_left + t_right)
    
    x_new = x + t * vx
    y_new = y + t * vy
    
    return x_new, y_new, t

def compute_phi_from_point(x, y):
    """Calcula el ángulo polar phi para un punto (x,y)."""
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    return phi

def billiard_trajectory(x0, y0, vx0, vy0, epsilon, num_reflections):
    """Calcula la trayectoria de una partícula en el billar."""
    # Normalizar velocidad inicial
    v_norm = np.sqrt(vx0**2 + vy0**2)
    if v_norm < 1e-12:
        raise ValueError("Velocidad inicial es cero")
    
    vx, vy = vx0 / v_norm, vy0 / v_norm
    x, y = x0, y0
    xs, ys = [x], [y]
    
    for i in range(num_reflections):
        try:
            # Encontrar siguiente intersección
            x_new, y_new, dt = find_next_intersection(x, y, vx, vy, epsilon)
            
            # Verificar que el paso es razonable
            if dt < 1e-10 or dt > 50:
                break
            
            # Calcular el ángulo correspondiente al punto de intersección
            phi = compute_phi_from_point(x_new, y_new)
            
            # Obtener vector normal en el punto de intersección
            nx, ny = normal_vector(phi, epsilon)
            
            # Verificar que la normal es válida
            norm_n = np.sqrt(nx**2 + ny**2)
            if norm_n < 1e-12:
                break
            
            # Reflejar la velocidad
            vx, vy = reflect_velocity(vx, vy, nx, ny)
            
            # Verificar que la velocidad reflejada es válida
            v_norm_new = np.sqrt(vx**2 + vy**2)
            if v_norm_new < 1e-12:
                break
            
            # Actualizar posición
            x, y = x_new, y_new
            xs.append(x)
            ys.append(y)
            
        except (ValueError, RuntimeError):
            break
    
    return np.array(xs), np.array(ys)

def calculate_lyapunov_exponent(x0, y0, vx0, vy0, epsilon, num_reflections=40, delta=1e-6):
    """Calcula el exponente de Lyapunov para una condición inicial dada."""
    try:
        # Trayectorias original y perturbada
        xs1, ys1 = billiard_trajectory(x0, y0, vx0, vy0, epsilon, num_reflections)
        xs2, ys2 = billiard_trajectory(x0 + delta, y0, vx0, vy0, epsilon, num_reflections)
        
        n = min(len(xs1), len(xs2))
        if n < 10:  # Necesitamos suficientes puntos
            return None
        
        # Calcular distancias logarítmicas
        log_distances = []
        for i in range(1, n):  # Empezar desde 1 para evitar log(0)
            d = np.sqrt((xs1[i] - xs2[i])**2 + (ys1[i] - ys2[i])**2)
            if d > delta:  # Solo considerar cuando la separación es significativa
                log_distances.append(np.log(d))
        
        if len(log_distances) < 5:
            return None
        
        # Ajuste lineal en la región apropiada
        reflections = np.arange(len(log_distances))
        start_idx = min(2, len(log_distances) // 4)
        end_idx = min(len(log_distances), max(10, len(log_distances) * 3 // 4))
        
        if end_idx - start_idx < 3:
            return None
        
        # Realizar regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(
            reflections[start_idx:end_idx], 
            log_distances[start_idx:end_idx]
        )
        
        # Solo aceptar resultados con buena correlación
        if r_value**2 > 0.3 and not np.isnan(slope):
            return max(slope, 0)  # El exponente de Lyapunov debe ser no negativo
        else:
            return None
            
    except Exception as e:
        return None

def generate_random_initial_conditions(n_conditions, epsilon):
    """Genera condiciones iniciales aleatorias dentro del billar."""
    conditions = []
    max_attempts = n_conditions * 5  # Limitar intentos
    attempts = 0
    
    while len(conditions) < n_conditions and attempts < max_attempts:
        attempts += 1
        
        # Generar punto aleatorio dentro del billar
        r_max = 1 + epsilon
        
        # Método de rechazo para puntos uniformes dentro del billar
        for _ in range(50):  # Máximo 50 intentos por punto
            # Generar punto en el círculo que contiene al billar
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 0.8 * (1 + epsilon * np.cos(angle)))
            
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            # Verificar si está dentro del billar
            r_check = np.sqrt(x**2 + y**2)
            if r_check > 1e-10:
                phi_check = np.arctan2(y, x)
                r_boundary = billiard_boundary(phi_check, epsilon)
                if r_check < 0.85 * r_boundary:  # Margen de seguridad
                    # Generar velocidad aleatoria
                    v_angle = np.random.uniform(0, 2 * np.pi)
                    vx = np.cos(v_angle)
                    vy = np.sin(v_angle)
                    
                    conditions.append((x, y, vx, vy))
                    break
    
    return conditions

def analyze_lyapunov_vs_epsilon(n_epsilon=30, n_conditions=20):
    """
    Calcula el exponente de Lyapunov promedio para diferentes valores de epsilon.
    """
    # Configurar estilo de matplotlib
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Valores de epsilon a evaluar (evitar epsilon = 1.0 exacto para estabilidad)
    epsilons = np.linspace(0.0, 0.95, n_epsilon)
    lyapunov_means = []
    lyapunov_stds = []
    
    print(f"Calculando exponentes de Lyapunov para {n_epsilon} valores de epsilon...")
    print(f"Usando {n_conditions} condiciones iniciales por valor de epsilon")
    print("-" * 60)
    
    for i, eps in enumerate(epsilons):
        print(f"Progreso: {i+1:2d}/{n_epsilon} (ε = {eps:.3f})", end="")
        
        # Generar condiciones iniciales aleatorias
        initial_conditions = generate_random_initial_conditions(n_conditions, eps)
        
        if len(initial_conditions) == 0:
            lyapunov_means.append(0)
            lyapunov_stds.append(0)
            print(" - Sin condiciones iniciales válidas")
            continue
        
        # Calcular exponentes de Lyapunov
        lyap_values = []
        for ic in initial_conditions:
            lyap = calculate_lyapunov_exponent(*ic, eps)
            if lyap is not None and not np.isnan(lyap):
                lyap_values.append(lyap)
        
        if lyap_values:
            mean_lyap = np.mean(lyap_values)
            std_lyap = np.std(lyap_values) if len(lyap_values) > 1 else 0
            lyapunov_means.append(mean_lyap)
            lyapunov_stds.append(std_lyap)
            print(f" - λ = {mean_lyap:.4f} ± {std_lyap:.4f} ({len(lyap_values)} válidos)")
        else:
            lyapunov_means.append(0)
            lyapunov_stds.append(0)
            print(" - Sin datos válidos")
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Convertir a arrays numpy
    lyapunov_means = np.array(lyapunov_means)
    lyapunov_stds = np.array(lyapunov_stds)
    
    # Graficar con barras de error
    ax.errorbar(epsilons, lyapunov_means, yerr=lyapunov_stds,
                fmt='o-', linewidth=2, markersize=6,
                capsize=4, capthick=1.5,
                color='#e74c3c', markerfacecolor='white', 
                markeredgecolor='#e74c3c', markeredgewidth=1.5,
                ecolor='#c0392b', alpha=0.8,
                label='Exponente de Lyapunov')
    
    # Línea de referencia en λ = 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Etiquetas y título
    ax.set_xlabel('ε')
    ax.set_ylabel('λ')
    ax.set_title('Coeficiente de Lyapunov en función de ε', fontweight='bold', pad=20)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    
    # Ajustar límites
    max_val = np.max(lyapunov_means[lyapunov_means > 0]) if np.any(lyapunov_means > 0) else 0.1
    ax.set_ylim(-0.02, max_val * 1.1)
    ax.set_xlim(-0.02, 1.0)
    
    plt.tight_layout()
    return fig, epsilons, lyapunov_means, lyapunov_stds

# Ejecutar el análisis
if __name__ == "__main__":
    print("Iniciando análisis del exponente de Lyapunov vs epsilon...")
    print("=" * 60)
    
    # Realizar el análisis
    fig, epsilons, means, stds = analyze_lyapunov_vs_epsilon(n_epsilon=25, n_conditions=30)
    
    # Guardar la figura
    plt.savefig('lyapunov_vs_epsilon.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("\n" + "=" * 60)
    print("¡Análisis completado!")
    print("Figura guardada como 'lyapunov_vs_epsilon.png'")
    

    
    # Mostrar estadísticas finales
    valid_means = means[means > 0]
    if len(valid_means) > 0:
        max_idx = np.argmax(means)
        print(f"\nExponente de Lyapunov máximo: {means[max_idx]:.4f} en ε = {epsilons[max_idx]:.3f}")
        
        # Encontrar transición aproximada al caos
        chaos_indices = np.where(means > 0.05)[0]
        if len(chaos_indices) > 0:
            print(f"Transición aproximada al caos: ε ≈ {epsilons[chaos_indices[0]]:.3f}")
    else:
        print("\nNo se encontraron exponentes de Lyapunov positivos significativos")
    
    plt.show()