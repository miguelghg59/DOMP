import numpy as np
import time

"""  
    Algoritmo de búsqueda por entornos variables para el DOMP basado en Domínguez-Marín et al. (2003).
    
    Recibe siempre 3 argumentos de entrada: la matriz de costes, el vector de pesos, y el numero de 
    ubicaciones a abrir. De manera opcional, se puede modificar la distancia máxima de agitación
    (k_max) y el número máximo de iteraciones (max_iters).
    
    Devuelve un diccionario con la mejor solución obtenida, el valor objetivo alcanzado, el tiempo
    de ejecución y "log", un diccionario donde se van almacenanando las mejoras en la funcion objetivo 
    asi como la iteracion y el momento en el que ocurren.
    """

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
    
    # Obtenemos qué instalación es la más cercana para cada cliente
    loc_1 = solution[loc_1_idx] 
    
    # Costos actuales de d1 y d2
    c1 = sub_matrix[np.arange(n), loc_1_idx]
    c2 = sub_matrix[np.arange(n), loc_2_idx]
    return loc_1, c1, c2

def calculate_domp_cost(solution, cost_matrix, lambda_weights):
    min_costs = np.min(cost_matrix[:, solution], axis=1)
    return np.dot(np.sort(min_costs), lambda_weights)

def local_search_fast_interchange(current_solution, n, cost_matrix, lambda_weights, 
                                  global_best_cost=None, log=None, ti=None, iteration=None):
    """
    Búsqueda Local optimizada con Fast Interchange,  registrando mejoras intermedias en tiempo real.
    """
    best_sol = current_solution.copy()
    best_cost = calculate_domp_cost(best_sol, cost_matrix, lambda_weights)
    p = len(best_sol)
    
    improvement = True
    while improvement:
        improvement = False    
        # 1. Precalculamos el estado de distancias (d1, d2) de la solución actual
        loc_1, c1, c2 = get_d1_d2(best_sol, cost_matrix)
        
        # vemos las ubicaciones sin usar; estas serán candidatas a entrar en la solucion
        mask = np.ones(n, dtype=bool)
        mask[best_sol] = False #las no disponibles a entrar, pues ya estan en nuestra solucion actual.
        unused_facilities = np.where(mask)[0] # [0] para que el array que devuelve sea unidimensional
        
        for f_in in unused_facilities:
            c_in = cost_matrix[:, f_in] #Vector de costos de la instalación que entra
            
            for i in range(p): 
                f_out = best_sol[i] # la candidata a salir            
                # 2. LÓGICA FAST INTERCHANGE VECTORIZADA
                is_f_out_d1 = (loc_1 == f_out) # vemos para que clientes, la ubicacion candidata a salir era la más cercana.
                # si es la mas cercana, el nuevo coste es min(c_in,c2); sino min(c_in,c1)
                new_costs = np.where(is_f_out_d1, 
                                     np.minimum(c_in, c2), 
                                     np.minimum(c_in, c1))
                new_cost = np.dot(np.sort(new_costs), lambda_weights) #actualizamos el valor de la f.objetivo
                
                if new_cost < best_cost: #en caso de ser mejor, actualizamos la solucion de referencia
                    best_cost = new_cost
                    best_sol[i] = f_in
                    improvement = True
                    # --- LÓGICA REGISTRO INTERMEDIO --- Solo registra si log existe y bate el récord global
                    if log is not None and global_best_cost is not None:
                        if best_cost < global_best_cost:
                            global_best_cost = best_cost # Actualizamos la cota local a batir
                            log.append({"Valor": best_cost,
                                "Iteraciones": iteration, 
                                "Tiempo": time.time() - ti, 
                            })
                    # ---------------------------------------------
                    break #rompemos el bucle interior (nuestra solucion ha mejorado)                   
            if improvement: 
                break # Rompemos el bucle exterior para recalcular d1 y d2 con la nueva solución. Si no ha habido mejora, finaliza.
    return best_sol, best_cost

def vns_domp(cost_matrix, lambda_weights, p, k_max = 3, max_iters = 2):
    """
    Algoritmo VNS con intercambio rápido.
    """
    ti = time.time()
    n = cost_matrix.shape[0] 
    rng = np.random.default_rng()

    # 1. Solución Inicial
    current_solution = rng.choice(n, p, replace=False)
    best_cost = calculate_domp_cost(current_solution, cost_matrix, lambda_weights)
    log = [{"Valor":best_cost,"Iter":0,"Tiempo":time.time()-ti}]

    iteration = 1
    while iteration < max_iters:
        k = 1
        while k <= k_max:
            # 2. Shaking/Agitacion
            shaken_solution = current_solution.copy()
            
            indices_to_replace = rng.choice(p, k, replace=False)
            
            mask = np.ones(n, dtype=bool)
            mask[current_solution] = False
            candidates = np.where(mask)[0]
            
            new_facilities = rng.choice(candidates, k, replace=False)
            shaken_solution[indices_to_replace] = new_facilities
            
            # 3. Búsqueda Local con FAST INTERCHANGE (y almacenando informacion en el log)
            new_solution, new_cost = local_search_fast_interchange(
                shaken_solution, n, cost_matrix, lambda_weights,
                global_best_cost=best_cost, log=log, ti=ti, iteration=iteration
            )

            # 4. Decisión de Movimiento
            if new_cost < best_cost:
                current_solution = new_solution.copy()
                best_cost = new_cost
                k = 1  #como estamos en una solucion mejor que la que teniamos de referencia, reseteamos la distancia de shaking
            else:
                k += 1 
                
        iteration += 1
    
    tf = time.time()
    t = round(tf - ti, 3)
    return np.sort(current_solution), best_cost, t,log