import numpy as np
import random
import time

def ep(cost_matrix, lambda_vec, p, pop_size=300, generations=1800, p_mutation=0.125, p_crossover=0.1):
    """
    Algoritmo Evolutivo para el DOMP basado en Domínguez-Marín et al. (2003).
    
    Recibe siempre 3 argumentos de entrada: la matriz de costes, el vector de pesos, y el numero de 
    ubicaciones a abrir. De manera opcional, se puede modificar el tamaño de la población, el número de
    generaciones, la probabilidad de que haya mutación y la probabilidad de cruce.
    
    Devuelve un diccionario con la mejor solución obtenida, el valor objetivo alcanzado, el tiempo
    de ejecución y "log", un diccionario donde se van almacenanando las mejoras en la funcion objetivo 
    asi como la iteracion y el momento en el que ocurren.
    """
    # esas probabilidades de mutacion y de cruce son las que destacan en el articulo como mas robustas
    ti = time.time() #empieza a contar el tiempo
    n = cost_matrix.shape[0]
    
    def evaluate(individual):
        # Para cada cliente, encontrar el costo mínimo a una de las instalaciones abiertas y evaluar la OMf
        costs = np.min(cost_matrix[:, individual], axis=1)
        sorted_costs = np.sort(costs)
        return np.dot(lambda_vec, sorted_costs)

        # Generar individuo aleatorio para la poblacion inicial
    def get_random_individual():
        ind = random.sample(range(n), p)
        ind.sort()
        return ind

    def crossover(parent1, parent2):
        # Operador de cruce adaptado para mantener factibilidad 
        p1, p2 = set(parent1), set(parent2)
        common = p1.intersection(p2)
        diff1 = sorted(list(p1 - common))
        diff2 = sorted(list(p2 - common))
        
        if not diff1: return parent1, parent2 # si ambos padres son iguales, los hijos son iguales
        
        break_pos = random.randint(0, len(diff1) - 1) #si sale 0, tendriamos los mismos hijo que padres.
        # Intercambio de índices no comunes
        child1_diff = diff1[:break_pos] + diff2[break_pos:]
        child2_diff = diff2[:break_pos] + diff1[break_pos:]
        
        child1 = sorted(list(common) + child1_diff)
        child2 = sorted(list(common) + child2_diff)
        return child1[:p], child2[:p]

    def mutate(individual):
        # Intercambio de una instalación abierta por una cerrada 
        res = list(individual)
        pos_to_remove = random.randint(0, p - 1)
        possible_new = list(set(range(n)) - set(res))
        res[pos_to_remove] = random.choice(possible_new)
        res.sort()
        return res

    # 1. Inicialización (Población aleatoria) 
    population = [get_random_individual() for _ in range(pop_size)] #generar poblacion aleatoria
    log = []
    for gen in range(generations):
        new_individuals = []
        
        # 2. Aplicar Cruce
        for i in range(0, pop_size, 2):
            if random.random() < p_crossover and i+1 < pop_size:
                c1, c2 = crossover(population[i], population[i+1])
                new_individuals.extend([c1, c2])
        
        # 3. Aplicar Mutación
        for i in range(pop_size):
            if random.random() < p_mutation:
                new_individuals.append(mutate(population[i]))
        
        # 4. Selección de los H mejores individuos 
        combined = population + new_individuals
        combined.sort(key=lambda x: evaluate(x)) # evaluar fitness segun funcion objetivo
        population = combined[:pop_size] #cortamos la lista ya ordenada 
        log.append({"Valor":evaluate(population[0]),"Iter":gen,"Tiempo":time.time()-ti})
        
    best_sol = population[0]
    best_val = evaluate(best_sol)
    tf = time.time() #paramos el tiempo
    t = round(tf - ti,5) 
    return best_sol, best_val, t,log