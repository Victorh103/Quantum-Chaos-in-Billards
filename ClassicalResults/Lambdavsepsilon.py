import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def billiard_boundary(phi, epsilon):
    """Define la frontera del billar parametrizada en coordenadas polares.
    
    Args:
        phi: ángulo en radianes
        epsilon: parámetro de deformación (0 para círculo, 1 para cardioide)
    
    Returns:
        radio: distancia desde el origen
    """
    return 1 + epsilon * np.cos(phi)

def boundary_point(phi, epsilon):
    """Calcula el punto en la frontera para un ángulo dado.
    
    Args:
        phi: ángulo en radianes
        epsilon: parámetro de deformación
    
    Returns:
        (x, y): coordenadas cartesianas del punto en la frontera
    """
    r = billiard_boundary(phi, epsilon)
    return r * np.cos(phi), r * np.sin(phi)

def boundary_derivative(phi, epsilon):
    """Calcula la derivada de la frontera respecto a phi.
    
    Args:
        phi: ángulo en radianes
        epsilon: parámetro de deformación
    
    Returns:
        dr_dphi: derivada del radio respecto a phi
    """
    return -epsilon * np.sin(phi)

def normal_vector(phi, epsilon):
    """Calcula el vector normal unitario en un punto de la frontera.
    
    Args:
        phi: ángulo en radianes
        epsilon: parámetro de deformación
    
    Returns:
        (nx, ny): componentes del vector normal unitario
    """
    r = billiard_boundary(phi, epsilon)
    dr_dphi = boundary_derivative(phi, epsilon)
    
    # Vector tangente
    tx = -r * np.sin(phi) + dr_dphi * np.cos(phi)
    ty = r * np.cos(phi) + dr_dphi * np.sin(phi)
    
    # Normalización del vector tangente
    norm_t = np.sqrt(tx**2 + ty**2)
    tx /= norm_t
    ty /= norm_t
    
    # Vector normal 
    nx, ny = -ty, tx
    
    return nx, ny

def tangent_vector(phi, epsilon):
    """Calcula el vector tangente unitario en un punto de la frontera.
    
    Args:
        phi: ángulo en radianes
        epsilon: parámetro de deformación
    
    Returns:
        (tx, ty): componentes del vector tangente unitario
    """
    r = billiard_boundary(phi, epsilon)
    dr_dphi = boundary_derivative(phi, epsilon)
    
    # Vector tangente
    tx = -r * np.sin(phi) + dr_dphi * np.cos(phi)
    ty = r * np.cos(phi) + dr_dphi * np.sin(phi)
    
    # Normalización
    norm_t = np.sqrt(tx**2 + ty**2)
    tx /= norm_t
    ty /= norm_t
    
    return tx, ty

def reflect_velocity(vx, vy, nx, ny):
    """Refleja el vector velocidad respecto a la normal.
    
    Args:
        vx, vy: componentes de la velocidad incidente
        nx, ny: componentes del vector normal unitario
    
    Returns:
        (vx_new, vy_new): componentes de la velocidad reflejada
    """
    # Producto escalar v·n
    v_dot_n = vx * nx + vy * ny
    
    # Velocidad reflejada: v' = v - 2(v·n)n
    vx_new = vx - 2 * v_dot_n * nx
    vy_new = vy - 2 * v_dot_n * ny
    
    return vx_new, vy_new

def implicit_boundary_function(x, y, epsilon):
    """Función implícita F(x,y) = 0 que define la frontera del billar.
    
    Args:
        x, y: coordenadas cartesianas
        epsilon: parámetro de deformación
    
    Returns:
        valor de F(x,y)
    """
    r = np.sqrt(x**2 + y**2)
    cos_phi = x / r if r > 0 else 1.0
    return r - (1 + epsilon * cos_phi)

def find_next_intersection(x, y, vx, vy, epsilon):
    """Encuentra la siguiente intersección de la trayectoria con la frontera.
    
    Args:
        x, y: posición actual
        vx, vy: velocidad actual (unitaria)
        epsilon: parámetro de deformación
    
    Returns:
        (x_new, y_new): coordenadas del punto de intersección
        t: tiempo hasta la intersección
    """
    def f(t):
        """F(x+t*vx, y+t*vy) para encontrar t donde la trayectoria cruza la frontera."""
        return implicit_boundary_function(x + t * vx, y + t * vy, epsilon)
    
    # Buscamos valores de t donde cambia el signo de f(t)
    t_min, t_max = 0.0001, 10.0  # Evitamos t=0 (posición actual)
    
    # Para billares no convexos, necesitamos buscar más cuidadosamente
    if epsilon >= 0.5:  # La cardioide se vuelve no convexa cerca de ε=0.5
        # Evaluamos la función en varios puntos para buscar cambios de signo
        t_values = np.linspace(t_min, t_max, 100)
        f_values = [f(t) for t in t_values]
        sign_changes = np.where(np.diff(np.signbit(f_values)))[0]
        
        if len(sign_changes) == 0:
            raise ValueError("No se encontró intersección con la frontera")
        
        # Tomamos el primer cambio de signo
        idx = sign_changes[0]
        t_min, t_max = t_values[idx], t_values[idx + 1]
    
    # Usamos brentq para encontrar el cero con precisión
    try:
        t = brentq(f, t_min, t_max)
    except ValueError:
        # Si brentq falla, intentamos con un intervalo más grande
        t_values = np.linspace(0.0001, 20.0, 200)
        f_values = [f(t) for t in t_values]
        sign_changes = np.where(np.diff(np.signbit(f_values)))[0]
        
        if len(sign_changes) == 0:
            raise ValueError("No se encontró intersección con la frontera")
        
        idx = sign_changes[0]
        t = brentq(f, t_values[idx], t_values[idx + 1])
    
    # Calculamos el punto de intersección
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
    """Calcula la trayectoria de una partícula en el billar.
    
    Args:
        x0, y0: posición inicial
        vx0, vy0: velocidad inicial (será normalizada)
        epsilon: parámetro de deformación
        num_reflections: número de reflexiones a calcular
    
    Returns:
        xs, ys: listas con las coordenadas de los puntos de reflexión
    """
    # Normalizar velocidad inicial
    v_norm = np.sqrt(vx0**2 + vy0**2)
    vx, vy = vx0 / v_norm, vy0 / v_norm
    
    x, y = x0, y0
    xs, ys = [x], [y]
    
    for _ in range(num_reflections):
        try:
            # Encontrar siguiente intersección
            x_new, y_new, _ = find_next_intersection(x, y, vx, vy, epsilon)
            
            # Calcular el ángulo correspondiente al punto de intersección
            phi = compute_phi_from_point(x_new, y_new)
            
            # Obtener vector normal en el punto de intersección
            nx, ny = normal_vector(phi, epsilon)
            
            # Reflejar la velocidad
            vx, vy = reflect_velocity(vx, vy, nx, ny)
            
            # Actualizar posición
            x, y = x_new, y_new
            xs.append(x)
            ys.append(y)
        except ValueError as e:
            print(f"Error en la iteración: {e}")
            break
    
    return xs, ys

def plot_single_billiard_no_axes(epsilon, num_reflections, x0=0.5, y0=0.1, vx0=0.7, vy0=0.7, save_path=None):
    """
    Crea una visualización del billar sin ejes, leyendas, ni títulos.
    
    Args:
        epsilon: Valor del parámetro de deformación (0 para círculo, 1 para cardioide)
        num_reflections: Número de reflexiones a calcular
        x0, y0: Posición inicial
        vx0, vy0: Velocidad inicial
        save_path: Ruta para guardar la figura (opcional)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Configuración de estilo y tamaño
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['axes.facecolor'] = "#00ff59ff"
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Crear un mapa de colores personalizado
    main_color = np.array([231, 76, 60]) / 255.0  # Color base
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_gradient',
        [main_color * 0.6 + 0.4, main_color, main_color * 0.7]
    )
    
    # Dibujar el contorno del billar
    phi = np.linspace(0, 2 * np.pi, 1000)
    x_boundary, y_boundary = zip(*[boundary_point(p, epsilon) for p in phi])
    ax.plot(x_boundary, y_boundary, color='#1292c4', linewidth=2.5, zorder=2)
    ax.fill(x_boundary, y_boundary, color=main_color, alpha=0.08, zorder=1)
    
    # Calcular trayectoria
    xs, ys = billiard_trajectory(x0, y0, vx0, vy0, epsilon, num_reflections)
    colors = custom_cmap(np.linspace(0, 1, len(xs) - 1))
    line_widths = np.linspace(1.8, 0.8, len(xs) - 1)
    
    # Dibujar trayectoria
    for k in range(len(xs) - 1):
        ax.plot([xs[k], xs[k + 1]], [ys[k], ys[k + 1]], '-', 
                color=colors[k], linewidth=line_widths[k], 
                alpha=0.85, zorder=3)
    
    # Ocultar ejes, etiquetas y leyendas
    ax.axis('off')
    
    # Guardar si se proporciona una ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    
    return fig

if __name__ == "__main__":
    epsilon = 0
    num_reflections = 100
    x0, y0 = 0.5, 0.1
    vx0, vy0 = 0.7, 0.7
    
    fig = plot_single_billiard_no_axes(
        epsilon=epsilon,
        num_reflections=num_reflections,
        x0=x0, y0=y0, vx0=vx0, vy0=vy0,
        save_path=f'billar_epsilon_{epsilon:.2f}_n_{num_reflections}_no_axes.png'
    )
    plt.show()