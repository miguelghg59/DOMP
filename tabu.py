import numpy as np
from collections import deque
import time

def calculate_domp_cost(solution, cost_matrix, lambda_weights):
    min_costs = np.min(cost_matrix[:, solution], axis=1)
    return np.dot(np.sort(min_costs), lambda_weights)

def get_d1_d2(solution, cost_matrix):
    """
    - loc_1: El índice de la instalación que proporciona la distancia mínima a cada cliente.
    - c1: El vector de costos a la instalación más cercana.
    - c2: El vector de costos a la segunda instalación más cercana. """

    n = cost_matrix.shape[0]
    sub_matrix = cost_matrix[:, solution] # Extraemos solo las columnas de las instalaciones abiertas
    
    # Ordenamos los índices de menor a mayor costo para cada cliente (fila)
    idx = np.argsort(sub_matrix, axis=1)
    loc_1_idx = idx[:, 0]
    loc_2_idx = idx[:, 1]
    
    # Obtenemos qué instalación exacta es la más cercana para cada cliente (para saber si la quitamos luego)
    loc_1 = solution[loc_1_idx] 
    
    # Costos actuales de d1 y d2
    c1 = sub_matrix[np.arange(n), loc_1_idx]
    c2 = sub_matrix[np.arange(n), loc_2_idx]
    return loc_1, c1, c2

def tabu_search_domp(cost_matrix, lambda_weights, p, max_iters=120, tabu_size=5):
    "importante que el tabu_size sea menor que n-p, porque si no nos quedaremos sin movimientos disponibles"
    ti = time.time()
    n = cost_matrix.shape[0]
    rng = np.random.default_rng()
    
    # 1. Solución Inicial (Aleatoria para empezar)
    current_solution = rng.choice(n, p, replace=False)
    best_sol = current_solution.copy()
    current_cost = calculate_domp_cost(current_solution, cost_matrix, lambda_weights)
    best_cost = current_cost
    log = [{"Valor":best_cost,"Iter":0,"Tiempo":time.time()-ti}]
    
    # Gestión  de la lista tabú: FIFO con deque (first in first out)
    tabu_queue = deque(maxlen=tabu_size)
    tabu_set = set()
    iters = 1   
    
    while iters < max_iters:
        # Precalculamos el estado de la solución actual (Fast Interchange)
        loc_1, c1, c2 = get_d1_d2(current_solution, cost_matrix)
        mask = np.ones(n, dtype=bool)
        mask[current_solution] = False
        unused_facilities = np.where(mask)[0]
        
        best_cost_neighbor = np.inf
        
        for f_in in unused_facilities: #iteramos sobre candidatas a entrar
            c_in = cost_matrix[:, f_in]
            if f_in in tabu_set:
                continue #no rompemos el bucle, sino que terminamos esta iteración: no podemos meter esa ubicacion pues está en tabú.
            for f_out in current_solution: #candidatas a salir de nuestra solucion actual
                
                is_f_out_d1 = (loc_1 == f_out) # vemos para que clientes, la ubicacion candidata a salir era la más cercana.
                new_costs = np.where(is_f_out_d1, 
                                     np.minimum(c_in, c2), 
                                     np.minimum(c_in, c1))
                new_cost = np.dot(np.sort(new_costs), lambda_weights)
                
                if new_cost < best_cost_neighbor: # Buscamos el MEJOR del vecindario (aunque sea peor que el actual)
                    best_cost_neighbor = new_cost
                    f_out_chosen = f_out
                    f_in_chosen = f_in
        
        # Actualizar solución actual (hacemos el intercambio)
        current_solution = current_solution.copy()
        current_solution = np.delete(current_solution, np.where(current_solution == f_out_chosen))
        current_solution = np.append(current_solution, f_in_chosen)
        current_cost = best_cost_neighbor
        
        # Gestionar memoria Tabú
        if len(tabu_queue) == tabu_size:
            old_tabu = tabu_queue.popleft() #eliminamos el movimiento más antiguo
            tabu_set.discard(old_tabu)         
        tabu_queue.append(f_out_chosen)
        tabu_set.add(f_out_chosen)
        
        # Comprobar si es un nuevo récord global
        if current_cost < best_cost:
            best_cost = current_cost
            best_sol = current_solution.copy()
            log.append({"Valor":best_cost,"Iter":iters,"Tiempo":time.time()-ti})
            
        iters += 1
        
    tf = time.time()
    t= tf-ti    
    return np.sort(best_sol), best_cost,t,log