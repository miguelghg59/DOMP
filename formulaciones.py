from gurobipy import Model,GRB, quicksum
import numpy as np

'''Todas las funciones reciben 3 argumentos de entrada: el número de ubicaciones a abrir (p),
el vector de pesos de la función objetivo, y la matriz con los costes de asignación. 

Todas devuelven un diccionario donde se almacena, el código del estatus del modelo (si se alcanza
o no optimalidad), el valor objetivo alcanzado, el tiempo de ejecución, gap, el número de iteraciones
realizadas y el modelo (en caso de querer acceder a otros aspectos adicionales como el valor de las variables).
'''

def LDOMP1(p,pesos, costes):
    ejmod = Model(name="LDOMP1")
    n = costes.shape[0] 

    # Variables
    x = ejmod.addVars(n, vtype=GRB.BINARY, name="x") 
    z = ejmod.addVars(n, n, n, vtype=GRB.BINARY, name="z")

    # Restricciones con quicksum
    ejmod.addConstrs(quicksum(z[i, k, j] for i in range(n) for j in range(n)) == 1 for k in range(n))
    ejmod.addConstrs(quicksum(z[i, k, j] for k in range(n) for j in range(n)) == 1 for i in range(n)) 
    ejmod.addConstrs(
        (quicksum(costes[k, j] * z[i, k, j] for k in range(n) for j in range(n)) <= 
         quicksum(costes[k, j] * z[i+1, k, j] for k in range(n) for j in range(n)) 
         for i in range(n-1)), 
        name="monotonia_posiciones"
    )
    
    ejmod.addConstr(quicksum(x[j] for j in range(n)) == p, name="apertura")
    
    ejmod.addConstrs(
        x[j] >= quicksum(z[i, k, j] for i in range(n)) 
        for k in range(n) for j in range(n)
    )

    # Funcion objetivo con quicksum
    fobj = quicksum(pesos[i] * costes[k, j] * z[i, k, j] for i in range(n) for k in range(n) for j in range(n))
    ejmod.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE DE PARAMETROS
    ejmod.setParam('TimeLimit', 60) #tiempo limite
    ejmod.setParam('OutputFlag', 0) 
    
    # OPTIMIZAR
    ejmod.optimize()
    
    # --- EXTRACCIÓN DE RESULTADOS ---   
    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
    # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if ejmod.SolCount > 0:
        obj_val = ejmod.ObjVal
    
    # 2. Tiempo de ejecución
    runtime = ejmod.Runtime
    
    # 3. Gap de optimalidad (MIPGap)
    gap = ejmod.MIPGap

    return {
        "status": ejmod.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":ejmod.NodeCount,
        "iteraciones":ejmod.IterCount,
        "model": ejmod #Por si quieres consultar variables después
    }

def LDOMP2(p,pesos, costes):

    ejmod = Model(name = "LDOMP2")

    n = costes.shape[0] 
    c = [sorted(costes[j,])[-p] for j in range(n)] 

    # Variables de decisión
    x = ejmod.addVars(n, vtype=GRB.BINARY, name="x") 
    y = ejmod.addVars(n, n, vtype=GRB.CONTINUOUS, name="y") 
    s = ejmod.addVars(n, n, vtype=GRB.BINARY, name="s")
    w = ejmod.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="w") 

    # Restricciones del conjunto N (Apertura y Asignación)
    ejmod.addConstr(quicksum(x[j] for j in range(n)) == p, name="apertura")
    ejmod.addConstrs((quicksum(y[i, j] for j in range(n)) == 1 for i in range(n)), name="asignaciones")
    ejmod.addConstrs((y[i, j] <= x[j] for i in range(n) for j in range(n)), name="asign_abierta")

    # Restricciones del conjunto P (Permutación/Ordenación)
    ejmod.addConstrs((quicksum(s[i, j] for i in range(n)) == 1 for j in range(n)), name="perm_filas")
    ejmod.addConstrs((quicksum(s[i, j] for j in range(n)) == 1 for i in range(n)), name="perm_cols")

    # Restricciones adicionales 
    ejmod.addConstrs((w[i] <= w[i+1] for i in range(n-1)), name="monotonicidad")
    ejmod.addConstr(
        quicksum(w[i] for i in range(n)) == 
        quicksum(costes[k, j] * y[k, j] for k in range(n) for j in range(n)), 
        name="conservacion_costes"
    )

    ejmod.addConstrs(
        (w[i] >= quicksum(costes[k, j] * y[k, j] for j in range(n)) - c[k] * (1 - s[i, k]) 
         for i in range(n) for k in range(n)), 
        name="rara_ordenacion"
    )

    # Función objetivo
    fobj = quicksum(pesos[i] * w[i] for i in range(n))
    ejmod.setObjective(fobj, GRB.MINIMIZE)

    # AJUSTE DE PARAMETROS
    ejmod.setParam("OutputFlag", False)
    ejmod.setParam("TimeLimit",60)

    #OPTIMIZAR
    ejmod.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if ejmod.SolCount > 0:
        obj_val = ejmod.ObjVal

    # 2. Tiempo de ejecución
    runtime = ejmod.Runtime
        
    # 3. Gap de optimalidad 
    gap = ejmod.MIPGap

    return {
        "status": ejmod.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":ejmod.NodeCount,
        "iteraciones":ejmod.IterCount,
        "model": ejmod 
    }

def MDOMPOT(p,pesos,costes):
    #CREACION DEL MODELO
    modelo = Model(name = "MDOMP(OT)")
    n = costes.shape[0] # shape son las dimensiones de los arrays
    Q = len(set(pesos)) # numero de bloques (por estar en el caso monotono)
    lambda_q = [pesos[0]]+[pesos[i] for i in range(1,n) if pesos[i] != pesos[i-1]] #representantes de cada bloque
    delta_q = [lambda_q[0]]+[lambda_q[i]-lambda_q[i-1]for i in range(1,Q)] #saltos entre bloque

    def conteobloq(a):
        contador = 1
        current = a[0] 
        sol = []
        for i in range(1,len(a)):
            if a[i] != a[i-1]:
                sol.append(contador)
                contador = 1
                current = a[i]
            else:
                contador += 1
        sol.append(contador)
        return sol
    q = conteobloq(pesos) # numero de elementos por bloque          


    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.CONTINUOUS, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.BINARY, name = "y")
    t = modelo.addVars(Q,vtype = GRB.CONTINUOUS, lb = 0, name = "t")
    z = modelo.addVars(n,Q,vtype = GRB.CONTINUOUS, lb = 0, name = "z")

    # RESTRICCIONES conjunto X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")

    #restriccion adicional
    modelo.addConstrs(quicksum(costes[i,j]*x[i,j] for j in range(n))-t[k] <= z[i,k] for i in range(n) for k in range(Q))

    #FUNCIÓN OBJETIVO
    fobj = quicksum(delta_q[k]*(quicksum(q[j] for j in range(k,Q))*t[k]+quicksum(z[i,k] for i in range(n))) for k in range(Q))
    modelo.setObjective(fobj, GRB.MINIMIZE)

    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 300) 

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo 
    }

def MDOMPBHP(p,pesos,costes):

    modelo = Model(name = "MDOMP(BHP)")
    
    n = costes.shape[0] # shape son las dimensiones de los arrays
    Q = len(set(pesos)) # numero de bloques (por estar en el caso monotono)
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)] #saltos 
    K = [i for i in range(n) if delta[i]!= 0] #indices de salto no nulo
    indices = [(l,k) for k in K for l in range(k,n)]

    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.CONTINUOUS, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.BINARY, name = "y")
    u = modelo.addVars(indices,vtype = GRB.CONTINUOUS, lb = 0, name = "u")
    v = modelo.addVars(n,K,vtype = GRB.CONTINUOUS, lb = 0, name = "v")

    # RESTRICCIONES conjunto X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")

    #restriccion adicional
    modelo.addConstrs(u[l,k]+v[i,k] >= quicksum(costes[i,j]*x[i,j] for j in range(n)) for i in range(n) for (l,k) in indices)

    #FUNCIÓN OBJETIVO
    fobj = quicksum(delta[k]*(quicksum(u[l,k] for l in range(k,n))+quicksum(v[i,k] for i in range(n))) for k in K)
    modelo.setObjective(fobj, GRB.MINIMIZE)

    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 300) # estableciendo limite de tiempo

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad (MIPGap)
    # Gurobi lo devuelve como decimal (0.01 = 1%). Si no hay solución, suele ser infinito.
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo 
    }

def DOMPOTR1(p,pesos,costes):
    #CREACION DEL MODELO
    modelo = Model(name = "DOMP(OTR1)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    cg = np.unique(costes[costes != 0])
    g = len(cg)
    alpha = min(Kn) if Kn else n #por si estuviera vacía la lista
    
    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.BINARY, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.CONTINUOUS, name = "y")
    t = modelo.addVars(Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "t")
    z = modelo.addVars(n,Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "z")
    r = modelo.addVars(n,g, vtype =GRB.BINARY, name = "r")
    
    #RESTRICCIONES CONJUNTO X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")
    
    #RESTRICCIONES ESPECIFICAS
    modelo.addConstrs(t[k]+z[i,k] >= quicksum(costes[i,j]*x[i,j] for j in range(n)) for i in range(n) for k in Kp)
    modelo.addConstrs((quicksum(x[i,j] for j in range(n) if costes[i,j]>costes[i,m])+y[m] <= 1 for i in range(n) for m in range(n))
                      , name = "asgin_cercana")
    modelo.addConstrs(r[l-1,h] <= r[l,h] for l in range(alpha+1,n) for h in range(g))
    modelo.addConstrs(quicksum(r[l,h] for l in range(n)) == 
                      quicksum(x[i,j] for i in range(n) for j in range(n) if costes[i,j] >= cg[h]) for h in range(g))
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*((n-k)*t[k] + quicksum(z[i,k] for i in range(n))) for k in Kp) 
            +quicksum(delta[k]*(quicksum(r[l,g-1]*cg[g-1]+quicksum((r[l,h]-r[l,h+1])*cg[h] for h in range(g-1)) for l in range(k,n))) for k in Kn)
           ) #ojo que no es n-k+1
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 30) # estableciendo limite de tiempo

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo 
    }

def DOMPOTR2(p,pesos,costes):

    modelo = Model(name = "DOMP(OTR2)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    cg = np.unique(costes[costes != 0])
    g = len(cg)
    diff_c = [cg[0]] + [cg[h] - cg[h-1] for h in range(1, g)]
    
    c = [np.unique(fila[fila != 0]) for fila in costes] #unique los da ya ordenados. Se pueden pedir los indice originales
    G = [len(fila) for fila in c]
    diff_cih = [[fila[0]]+[fila[i]-fila[i-1] for i in range(1,len(fila))] for fila in c]
    
    l = [[min(s for s in range(G[i]) if c[i][s]>=cg[h]) if cg[h]<=c[i][G[i]-1] else G[i] for h in range(g)] for i in range(n) ] 
    indicesw = [(i,h) for i in range(n) for h in range(G[i])]
    
    #VARIABLES DE DECISION
    y = modelo.addVars(n,vtype = GRB.BINARY, name = "y")
    t = modelo.addVars(Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "t")
    z = modelo.addVars(n,Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "z")
    w = modelo.addVars(indicesw, vtype=GRB.CONTINUOUS, lb=0, ub = 1,name="w")
    rho = modelo.addVars(Kn, g, vtype=GRB.CONTINUOUS, lb=0, name="rho")
    
    #RESTRICCIONES
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p)
    modelo.addConstrs(w[i,h] >= 1-quicksum(y[j] for j in range(n) if costes[i,j]<c[i][h]) for (i,h) in indicesw)
    modelo.addConstrs((w[i,h]+y[j] <= 1 for i in range(n) for j in range(n) for h in range(G[i]) if costes[i,j]<c[i][h]),
                      name = "asign_cercana")
    modelo.addConstrs(t[k]+z[i,k] >= quicksum(w[i,h]*diff_cih[i][h] for h in range(G[i])) for i in range(n) for k in Kp)
    modelo.addConstrs(rho[k,h] <= quicksum(w[i,l[i][h]] for i in range(n) if l[i][h]< G[i]) for k in Kn for h in range(g))
    modelo.addConstrs(rho[k,h] <= n-k for k in Kn for h in range(g)) #no n-k+1
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*((n-k)*t[k] + quicksum(z[i,k] for i in range(n))) for k in Kp) 
            +quicksum(delta[k]*(quicksum(rho[k,h]*diff_c[h] for h in range(g))) for k in Kn)
           ) #ojo que no es n-k+1
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 30)

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo # Por si quieres consultar variables después
    }

def DOMPOTtheta(p,pesos,costes,tlim):

    modelo = Model(name = "DOMP(OTtheta)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    
    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.BINARY, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.CONTINUOUS, name = "y")
    t = modelo.addVars(Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "t")
    z = modelo.addVars(n,Kp,vtype = GRB.CONTINUOUS, lb = 0, name = "z")
    theta = modelo.addVars(n,n,Kn, vtype = GRB.CONTINUOUS, lb = 0, name = "theta")
    
    #RESTRICCIONES CONJUNTO X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")
    
    #RESTRICCIONES ADICIONALES
    modelo.addConstrs(t[k]+z[i,k] >= quicksum(costes[i,j]*x[i,j] for j in range(n)) for i in range(n) for k in Kp)
    modelo.addConstrs((quicksum(x[i,j] for j in range(n) if costes[i,j]>costes[i,m])+y[m] <= 1 for i in range(n) for m in range(n))
                      , name = "asgin_cercana") 
    modelo.addConstrs(quicksum(theta[i,j,k] for i in range(n) for j in range(n)) == n-k for k in Kn)
    modelo.addConstrs(theta[i,j,k] <= x[i,j] for i in range(n) for j in range(n) for k in Kn)
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*((n-k)*t[k] + quicksum(z[i,k] for i in range(n))) for k in Kp) 
            +quicksum(delta[k]*(quicksum(costes[i,j]*theta[i,j,k] for i in range(n) for j in range(n))) for k in Kn )
           ) #ojo que no es n-k+1
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", tlim) 

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo # Por si quieres consultar variables después
    }

def DOMPBHPR1(p,pesos,costes):
    #CREACION DEL MODELO
    modelo = Model(name = "DOMP(BHPR1)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    cg = np.unique(costes[costes != 0])
    g = len(cg)
    alpha = min(Kn) if Kn else n #por si estuviera vacía la lista
    indices_u = [(k,l) for k in Kp for l in range(k,n)]
    
    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.BINARY, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.CONTINUOUS, name = "y")
    u = modelo.addVars(indices_u,vtype = GRB.CONTINUOUS, lb = 0, name = "u")
    v = modelo.addVars(Kp,n,vtype = GRB.CONTINUOUS, lb = 0, name = "v")
    r = modelo.addVars(n,g, vtype =GRB.BINARY, name = "r")
    
    #RESTRICCIONES CONJUNTO X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")
    
    #RESTRICCIONES ESPECIFICAS
    modelo.addConstrs(u[k,l]+v[k,i] >= quicksum(costes[i,j]*x[i,j] for j in range(n))
                      for i in range(n) for k in Kp for l in range(k,n))
    
    modelo.addConstrs((quicksum(x[i,j] for j in range(n) if costes[i,j]>costes[i,m])+y[m] <= 1 for i in range(n) for m in range(n))
                      , name = "asgin_cercana")
    modelo.addConstrs(r[l-1,h] <= r[l,h] for l in range(alpha+1,n) for h in range(g))
    modelo.addConstrs(quicksum(r[l,h] for l in range(n)) == 
                      quicksum(x[i,j] for i in range(n) for j in range(n) if costes[i,j] >= cg[h]) for h in range(g))
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*(quicksum(u[k,l] for l in range(k,n)) + quicksum(v[k,i] for i in range(n))) for k in Kp) 
            +quicksum(delta[k]*(quicksum(r[l,g-1]*cg[g-1]+quicksum((r[l,h]-r[l,h+1])*cg[h] for h in range(g-1)) for l in range(k,n))) for k in Kn)
           ) #ojo que no es n-k+1
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 30) 

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo # Por si quieres consultar variables después
    }

def DOMPBHPR2(p,pesos,costes):
    #CREACION DEL MODELO
    modelo = Model(name = "MDOMP(BHPr2)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    cg = np.unique(costes[costes != 0])
    g = len(cg)
    diff_c = [cg[0]] + [cg[h] - cg[h-1] for h in range(1, g)]
    
    c = [np.unique(fila[fila != 0]) for fila in costes]
    G = [len(fila) for fila in c]
    diff_cih = [[fila[0]]+[fila[i]-fila[i-1] for i in range(1,len(fila))] for fila in c]
    
    l = [[min(s for s in range(G[i]) if c[i][s]>=cg[h]) if cg[h]<=c[i][G[i]-1] else G[i] for h in range(g)] for i in range(n) ] 
    indicesw = [(i,h) for i in range(n) for h in range(G[i])]
    
    indices_u = [(k,l) for k in Kp for l in range(k,n)]
    
    #VARIABLES DE DECISION
    y = modelo.addVars(n,vtype = GRB.BINARY, name = "y")
    u = modelo.addVars(indices_u,vtype = GRB.CONTINUOUS, lb = 0, name = "u")
    v = modelo.addVars(Kp,n,vtype = GRB.CONTINUOUS, lb = 0, name = "v")
    w = modelo.addVars(indicesw, vtype=GRB.CONTINUOUS, lb=0, ub = 1,name="w")
    rho = modelo.addVars(Kn, g, vtype=GRB.CONTINUOUS, lb=0, name="rho")
    
    #RESTRICCIONES
    modelo.addConstrs(u[k,l]+v[k,i] >= quicksum(diff_cih[i][h]*w[i,h] for h in range(G[i]))
                      for i in range(n) for k in Kp for l in range(k,n))
    
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p)
    modelo.addConstrs(w[i,h] >= 1-quicksum(y[j] for j in range(n) if costes[i,j]<c[i][h]) for (i,h) in indicesw)
    modelo.addConstrs((w[i,h]+y[j] <= 1 for i in range(n) for j in range(n) for h in range(G[i]) if costes[i,j]<c[i][h]),
                      name = "asign_cercana")
    modelo.addConstrs(rho[k,h] <= quicksum(w[i,l[i][h]] for i in range(n) if l[i][h]< G[i]) for k in Kn for h in range(g))
    modelo.addConstrs(rho[k,h] <= n-k for k in Kn for h in range(g)) #no n-k+1
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*(quicksum(u[k,l] for l in range(k,n)) + quicksum(v[k,i] for i in range(n))) for k in Kp)
            +quicksum(delta[k]*(quicksum(rho[k,h]*diff_c[h] for h in range(g))) for k in Kn)
           ) 
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", 30) 

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad 
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo # Por si quieres consultar variables después
    }

def DOMPBHPtheta(p,pesos,costes,tlim):
    #CREACION DEL MODELO
    modelo = Model(name = "MDOMP(BHPtheta)")
    
    n = costes.shape[0]
    delta = [pesos[0]]+[pesos[i]-pesos[i-1]for i in range(1,n)]
    Kp = [i for i in range(n) if delta[i]>0]
    Kn = [i for i in range(n) if delta[i]<0]
    
    indices_u = [(k,l) for k in Kp for l in range(k,n)]
    #VARIABLES DE DECISION
    x = modelo.addVars(n,n,vtype = GRB.BINARY, lb = 0, name = "x")
    y = modelo.addVars(n,vtype = GRB.CONTINUOUS, name = "y")
    u = modelo.addVars(indices_u,vtype = GRB.CONTINUOUS, lb = 0, name = "u")
    v = modelo.addVars(Kp,n,vtype = GRB.CONTINUOUS, lb = 0, name = "v")
    theta = modelo.addVars(n,n,Kn, vtype = GRB.CONTINUOUS, lb = 0, name = "theta")
    
    #RESTRICCIONES CONJUNTO X
    modelo.addConstr(quicksum(y[j] for j in range(n)) == p, name = "aperturas")
    modelo.addConstrs((quicksum(x[i,j] for j in range(n)) == 1 for i in range(n)),name = "asignaciones")
    modelo.addConstrs((x[i,j]<= y[j] for i in range(n) for j in range(n)), name ="asign_abierta")
    
    #RESTRICCIONES ADICIONALES
    modelo.addConstrs((quicksum(x[i,j] for j in range(n) if costes[i,j]>costes[i,m])+y[m] <= 1 for i in range(n) for m in range(n))
                      , name = "asgin_cercana") 
    modelo.addConstrs(quicksum(theta[i,j,k] for i in range(n) for j in range(n)) == n-k for k in Kn)
    modelo.addConstrs(theta[i,j,k] <= x[i,j] for i in range(n) for j in range(n) for k in Kn)
    modelo.addConstrs(u[k,l]+v[k,i] >= quicksum(costes[i,j]*x[i,j] for j in range(n))
                      for i in range(n) for k in Kp for l in range(k,n))
    
    #FUNCIÓN OBJETIVO
    fobj = (quicksum(delta[k]*(quicksum(u[k,l] for l in range(k,n)) + quicksum(v[k,i] for i in range(n))) for k in Kp)
            +quicksum(delta[k]*(quicksum(costes[i,j]*theta[i,j,k] for i in range(n) for j in range(n))) for k in Kn )
           ) 
    modelo.setObjective(fobj, GRB.MINIMIZE)
    
    #AJUSTE PARAMETROS
    modelo.setParam("OutputFlag", False)
    modelo.setParam("TimeLimit", tlim) 

    #OPTIMIZACION
    modelo.optimize()

    # --- EXTRACCIÓN DE RESULTADOS ---

    # 1. Valor óptimo (o la mejor cota superior encontrada hasta el momento)
     # Verificamos si se encontró al menos una solución factible
    obj_val = None
    if modelo.SolCount > 0:
        obj_val = modelo.ObjVal

    # 2. Tiempo de ejecución
    runtime = modelo.Runtime
        
    # 3. Gap de optimalidad
    gap = modelo.MIPGap

    return {
        "status": modelo.Status,
        "obj_val": obj_val,
        "runtime": runtime,
        "gap": gap,
        "nodos":modelo.NodeCount,
        "iteraciones":modelo.IterCount,
        "model": modelo # Por si quieres consultar variables después
    }