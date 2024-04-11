
'''
Funciones auxiliares usadas en:
Branching process - etiquetar
Barabasi - etiquetar
Ejemplo multi-type branching process
'''


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import scipy

## -----------------------------------------------------------------------------------------------------------------------------

# DIBUJAR GRAFOS

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos

def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / 2
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
    return pos

def draw_tree(tree, root=0, colores = False):

    # si colores == True, significa que el arbol tiene colores asignados
    # como node atributes, con la llave de 'color'
    if colores:
        # tomar los atributos de los colores
        colores_nodos = [tree.nodes[nodo]['color'] for nodo in tree.nodes()]
    # sino tienen atributos de colores, ponerlos a todos del mismo color
    else:
        colores_nodos = "skyblue"

    # dibujar el arbol
    pos = hierarchy_pos(tree, root)
    nx.draw(tree, pos, with_labels=True, node_size=700, node_color=colores_nodos, font_size=8, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
    plt.show()

## -----------------------------------------------------------------------------------------------------------------------------

# GENERACION BRANCHING PROCESS

def sample_from_offspring():

    # uso una uniforme de 1 a 3, pero esto se cambia facilmente

    return np.random.choice([1, 2, 3])

def branching_process(numero_de_generaciones):
    # usa la funcion: sample_from_offspring
    # para crear un branching process con esa distribucion
    # donde se especifica el numero maximo de generaciones

    # crear el grafo
    grafo = nx.DiGraph()

    # hacer un diccionario de generaciones, es decir de pares
    # numero_de_generacion : [nombre de los nodos en esa generacion]
    dict_generaciones = dict()

    # los nodos tienen nombres de numeros
    # ir llevando esta variable
    # sumar en uno cada que se pone un nuevo nodo
    nombre_siguiente_nodo = 0

    # poner la 0-esima generacion
    grafo.add_node(nombre_siguiente_nodo)

    # indicarlo en el dissionario de generaciones
    dict_generaciones[0] = [nombre_siguiente_nodo]

    # sumar en uno el nombre del siguiente nodo
    nombre_siguiente_nodo = nombre_siguiente_nodo + 1

    # poner tantas generaciones como se quieran
    for n in range(1, numero_de_generaciones + 1):

        print(f"Creando la generacion: {n}")

        # indicar que en este generacion,
        # de entrada no se tienen nodos
        # inicializarlo de esta forma
        dict_generaciones[n] = []

        # iterar en los nodos de la generacion pasada
        for node_padre in dict_generaciones[n-1]:

            # por cada uno, ponerle un numero de hijos
            # de acuerdo a la offsrping distribution

            # ver cuantos hijos se van a poner
            num_hijos = sample_from_offspring()

            # poner cada uno de estos hijos
            for _ in range(num_hijos):

                # poner un nodo extra, que es el hijo
                grafo.add_node(nombre_siguiente_nodo)

                # poner la conexion de padre a hijo
                grafo.add_edge(node_padre, nombre_siguiente_nodo)

                # añadir al hijo a los nodos de esta generacion
                dict_generaciones[n].append(nombre_siguiente_nodo)

                # sumar en uno el nombre para el siguiente nodo
                nombre_siguiente_nodo = nombre_siguiente_nodo + 1


        print(f"\tNumero de nodos en esta generacion: {len(dict_generaciones[n])} ")


    # comprobar que sea un DAG
    assert nx.is_directed_acyclic_graph(grafo)

    # devolver el grafo y el diccionario de generaciones
    return grafo, dict_generaciones


## -----------------------------------------------------------------------------------------------------------------------------

# BARABASI

def convertir_a_arbol_dirigido(grafo_no_dirigido, nodo_raiz):

    # Encuentra la distancia de cada nodo al nodo raíz
    distancias = nx.single_source_shortest_path_length(grafo_no_dirigido, nodo_raiz)

    # Inicializa un grafo dirigido
    grafo_dirigido = nx.DiGraph()

    # Añade nodos al grafo dirigido
    grafo_dirigido.add_nodes_from(grafo_no_dirigido.nodes())

    # Añade aristas al grafo dirigido con dirección de nodo más cercano al nodo raíz a más lejano
    for nodo_destino, distancia in distancias.items():
        if nodo_destino != nodo_raiz:
            # Encuentra el nodo anterior en el camino hacia el nodo raíz
            nodo_anterior = min(grafo_no_dirigido.neighbors(nodo_destino), key=lambda x: distancias[x])

            # Añade una arista dirigida desde el nodo anterior al nodo destino
            grafo_dirigido.add_edge(nodo_anterior, nodo_destino)

    return grafo_dirigido


def arbol_barabasi(num_nodos):

    # usar las funciones para generar un arbol con barabasi
    arbol_no_dirigido = barabasi_albert_graph(num_nodos, 1)

    # hacerlo dirigido, la raiz es el 0
    arbol_dirigido = convertir_a_arbol_dirigido(arbol_no_dirigido, 0)

    # comprobar que sea un DAG
    assert nx.is_directed_acyclic_graph(arbol_dirigido)

    return arbol_dirigido

## -----------------------------------------------------------------------------------------------------------------------------

# CADENA DE MARKOV

def generate_random_transition_matrix(states):

    num_states = len(states)

    # Generate a random matrix with values between 0 and 1
    random_matrix = np.random.rand(num_states, num_states)

    # Normalize the matrix so that each row sums to 1
    transition_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)

    return transition_matrix

def visualize_markov_chain(states, transition_matrix):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph without labels
    for state in states:
        G.add_node(state, color=state)

    # Add edges with weights proportional to conditional probabilities
    for i in range(len(states)):
        for j in range(len(states)):
            weight = transition_matrix[i, j]
            if weight > 0:
                G.add_edge(states[i], states[j], weight=weight)

    # Extract node colors from the graph
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=1000, node_color=node_colors, font_size=10, font_color='white',
            font_weight='bold', edge_color='black', width=[edge[2]['weight'] * 10 for edge in G.edges(data=True)],
            arrowsize=20, connectionstyle='arc3,rad=0.1')  # Modified to draw edges between all pairs

    # Display the plot
    plt.show()

## -----------------------------------------------------------------------------------------------------------------------------

# ETIQUETAR ARBOL

# ponerle una etiqueta a un nodo en un grafo
def etiquetar(nodo, indice_etiqueta, grafo):

    # poner la etiqueta como parte del diccionario
    # asociado a cada nodo
    # con la llave de etiqueta
    grafo.nodes[nodo]['etiqueta'] = indice_etiqueta


# dado un nodo de un grafo, obtener su etiqueta
def get_etiqueta(nodo, grafo):

    # devuelve la etiqueta de un nodo
    return grafo.nodes[nodo]['etiqueta']


# funcion usada para escojer un indice de acuerdo a las probabilidades
def elegir_indice_con_probabilidades(probabilidades):

    indices = list(range(len(probabilidades)))

    indice_elegido = random.choices(indices, weights=probabilidades, k=1)[0]

    return indice_elegido



def etiquetar_arbol(arbol, nodo_raiz, indice_etiqueta_raiz, transition_matrix, etiquetas):

    # tomar los nods del arbol
    nodos = list(arbol.nodes)

    # poner una etiqueta por nodo
    # de momento todos inician en -1
    for nodo in nodos:
        etiquetar(nodo, -1, arbol)

    # la etiqueta de la raiz se define antes de empezar

    # iterar en los nodos del grafo para etiquetar
    # iterar usando bfs

    visitados = set()
    cola = [nodo_raiz]

    # iterar mientras haya elementos en la cola
    while cola:
        # sacar el primero de la cola (bfs)
        nodo_actual = cola.pop(0)

        # si no se ha visto
        if nodo_actual not in visitados:

            # se etiqueta el nodo

            # si es la raiz, ya se definio la etiquera
            if nodo_actual == nodo_raiz:
                etiquetar(nodo_actual, indice_etiqueta_raiz, arbol)

            # no es la raiz,
            # definir la etiqueta con base a la etiqeuta del padre
            else:

                # tomar los padres
                padres_ = arbol.predecessors(nodo_actual)
                padres = [p for p in padres_]

                # comprobar que sesa solo 1, y tomarlo
                assert len(padres) == 1

                padre = padres[0]

                # tomar el indice de la etiqueta del padre
                indice_etiqueta_padre = get_etiqueta(padre, arbol)

                # comprobar que ya tenga etiqueta
                assert indice_etiqueta_padre != -1

                # tomas las probabilidades
                proba_etiquetas = transition_matrix[indice_etiqueta_padre]

                # hacer un sampleo, segun estas probabilidades
                # para seleccionar el indice de esta etiqueta
                indice_etiqueta_actual = elegir_indice_con_probabilidades(proba_etiquetas)

                # etiquetar
                etiquetar(nodo_actual, indice_etiqueta_actual, arbol)


            # como sea, se marca como visitado
            visitados.add(nodo_actual)

            # se añadren los siguientes nodos
            vecinos = list(arbol.successors(nodo_actual))
            cola.extend(vecinos)

    # se tienen los indices de las etiquetas
    # ademas, se quieren los colores asociados

    # convertirlas en colores
    # tambien ponerle colores al grafo
    for nodo in nodos:
        arbol.nodes[nodo]['color'] = etiquetas[get_etiqueta(nodo, arbol)]

## -----------------------------------------------------------------------------------------------------------------------------

# DISTRIBUCION LIMITE

# comprueba que una matriz es una matriz de probabilidad
# i.e que las filas suman a 1, y no tiene entradas negativas
def comprobar_matriz_probabilidad(matriz):

    # verificar si todas las entradas son no negativas
    if np.any(matriz < 0):
        return False

    # verificar si la suma de cada fila es aproximadamente igual a 1
    filas_suman_a_1 = np.all(np.isclose(np.sum(matriz, axis=1), 1))

    return filas_suman_a_1

def get_distribucion_limite(matriz_transicion):
    '''
    Toma la matriz de transicion de una cadena de markov
    Devuelve la distribucion limite de la cadena de markov

    Lo que se hace es checar si es irreducible y aperiodica.
    Pues si la cadena de markov es irreducible, aperiodica.
    Entonces la distribucion limite es la distribucion estacionaria.
    '''

    # comprobar que si sea de probabilidad
    assert comprobar_matriz_probabilidad(matriz_transicion)

    # Crear un grafo dirigido desde la matriz de transición
    grafo_markov = nx.DiGraph(matriz_transicion)

    # ver si es irreducible
    irreducible = nx.is_strongly_connected(grafo_markov)

    if irreducible:
        print("La cadena es irreducible")
    else:
        print("La cadena no es irreducible")
        return None

    # ver si es aperiodica
    is_aperiodic = nx.is_aperiodic(grafo_markov)

    if irreducible:
        print("La cadena es aperiodica")
    else:
        print("La cadena no es aperiodica")
        return None

    # Calcular la dsitribucion estacionaria

    # eigenvalores y eigenvectores
    eigenvalores, eigenvectores = np.linalg.eig(matriz_transicion.T)

    # tomar el eigenvector asociado al eigenvalor 1
    eigenvector_1 = np.real_if_close(eigenvectores[:, np.isclose(eigenvalores, 1)])

    # reshape
    eigenvector_1 = eigenvector_1.reshape(-1)

    # si suma uno es la distribucion estacionaria
    distribucion_estacionaria = eigenvector_1 / np.sum(eigenvector_1)

    # devolverla
    return list(distribucion_estacionaria)

## -----------------------------------------------------------------------------------------------------------------------------

# COMPARAR DISTRIBUCIONES

def plot_distribution(distribucion, ax, etiquetas):
    # toma una ditribucion sobre las etiquetas
    # la grafica

    # ver que tenga los mismos elementos
    assert len(etiquetas)==len(distribucion)

    # hacer las barras
    ax.bar(etiquetas, distribucion)

    # poner los valores
    for i, valor in enumerate(distribucion):
        ax.text(i, valor, f'{valor:.3f}', ha='center', va='bottom')

    # devolevr el ax
    return ax

# funcion para calcular la KL divergence D(p|q)
# no es una metrica, no es simetrica ni lo del triangulo, es una divergencia
def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)

## -----------------------------------------------------------------------------------------------------------------------------

# ESTIMAR CADENA DE MARKOV

def estimar_markov(grafo, num_etiquetas):
    '''
    Toma un grafo etiquetado
    Es decir, donde cada nodo tiene un atributo con nombre 'etiqueta'
    Las etiquetas son numeros en {0, 1, ..., num_etiquetas-1}
    Se asume que el etiquetado del grafo sigue un proceso de cadena de markov
    Se busca estimar la matriz de transicion de esta
    Se devuelve la estimacion de esta matriz de transicion
    '''

    # hacer una matriz que cuente de donde a donde van las aristas
    # inicializar la matriz con ceros
    matriz_contar = np.zeros((num_etiquetas, num_etiquetas))

    # iterar en las aristas
    for edge in grafo.edges():

        # tomar al padre, y su etiqueta
        padre = edge[0]
        etiqueta_padre = get_etiqueta(padre, grafo)

        # tomar al hijo, y su etiqueta
        hijo = edge[1]
        etiqueta_hijo = get_etiqueta(hijo, grafo)


        # sumar en uno la cuenta
        # de las aristas que vienen de la etiqueta del padre
        # a la etiqueta del hijo
        matriz_contar[etiqueta_padre, etiqueta_hijo] = matriz_contar[etiqueta_padre, etiqueta_hijo] + 1

    # comprobar que se hayan contado todas las aristas del grafo
    assert grafo.number_of_edges() == matriz_contar.sum()

    # si una fila se llena de ceros
    # entonces no se tiene info de la proba condicional
    # se llena con unos, para que sea uniforme
    for fila in range(matriz_contar.shape[0]):
            if np.all(matriz_contar[fila, :] == 0):
                print(f"No se tenia informacion hijos cuyo padre tiene etiqueta: {fila}")
                print("La informacion condicional dada esa etiqueta se hace uniforme")
                matriz_contar[fila, :] = 1


    # para estimar las probabilidades condicionales
    # se normaliza cada fila para que sume uno
    matriz_estimada = matriz_contar / matriz_contar.sum(axis=1)[:, np.newaxis]

    # comprobar que sea matriz de probabilidad
    assert comprobar_matriz_probabilidad(matriz_estimada)

    # devolver la estimacion
    return matriz_estimada


## -----------------------------------------------------------------------------------------------------------------------------

# MULTI-TYPE BRANCHING PROCESS

def checar_distribucion(dict_probabilidades):

    # ver que las probabilidades sumen a 1
    suma_probabilidades = np.array(list(dict_probabilidades.values())).sum()
    return np.isclose(suma_probabilidades, 1)

def esperanza_distribucion(dict_probabilidades):

    # ver que si es distribucion
    assert checar_distribucion(dict_probabilidades)

    esperanza = 0

    # iterar en los pares
    for x, px in dict_probabilidades.items():
        # sumar el termino correspondiente
        esperanza = esperanza + x*px

    return esperanza

def sample_from_offspring_distribution(dict_probabilidades):

    # ver que si es distribucion
    assert checar_distribucion(dict_probabilidades)

    return random.choices(list(dict_probabilidades.keys()),
                          weights= list(dict_probabilidades.values()), k=1)[0]


def generar_arbol_etiquetado(numero_de_generaciones, offspring_distributions, transition_matrix, indice_etiqueta_raiz, etiquetas):
    # genera el arbol y lo va etiquetando

    # crear el grafo
    grafo = nx.DiGraph()

    # hacer un diccionario de generaciones, es decir de pares
    # numero_de_generacion : [nombre de los nodos en esa generacion]
    dict_generaciones = dict()

    # los nodos tienen nombres de numeros
    # ir llevando esta variable
    # sumar en uno cada que se pone un nuevo nodo
    nombre_siguiente_nodo = 0

    # poner la 0-esima generacion
    grafo.add_node(nombre_siguiente_nodo)

    # indicarlo en el diccionario de generaciones
    dict_generaciones[0] = [nombre_siguiente_nodo]

    # ponerle la etiquera correspondiente al nodo raiz
    etiquetar(nombre_siguiente_nodo, indice_etiqueta_raiz, grafo)

    # sumar en uno el nombre del siguiente nodo
    nombre_siguiente_nodo = nombre_siguiente_nodo + 1

    # poner tantas generaciones como se quieran
    for n in range(1, numero_de_generaciones + 1):

        print(f"Creando y etiquetando la generacion: {n}")

        # indicar que en este generacion,
        # de entrada no se tienen nodos
        # inicializarlo de esta forma
        dict_generaciones[n] = []

        # iterar en los nodos de la generacion pasada
        for node_padre in dict_generaciones[n-1]:

            # el numero de hijos, y tipo de estos,
            # dependen del tipo del padre
            tipo_padre = get_etiqueta(node_padre, grafo)

            # ver cual es la offspring distribution correspondiente
            # segun el tipo del padre
            offspring_padre = offspring_distributions[tipo_padre]

            # usar esta distribucion para dar el numero de hijos
            num_hijos = sample_from_offspring_distribution(offspring_padre)

            # poner cada uno de estos hijos
            for _ in range(num_hijos):

                # poner un nodo extra, que es el hijo
                grafo.add_node(nombre_siguiente_nodo)

                # poner la conexion de padre a hijo
                grafo.add_edge(node_padre, nombre_siguiente_nodo)

                # añadir al hijo a los nodos de esta generacion
                dict_generaciones[n].append(nombre_siguiente_nodo)

                # etiquetar al hijo
                # esto depende de la etiqueta del padre

                # tomas las probabilidades
                proba_etiquetas = transition_matrix[tipo_padre]

                # hacer un sampleo, segun estas probabilidades
                # para seleccionar el indice de esta etiqueta
                indice_etiqueta_actual = elegir_indice_con_probabilidades(proba_etiquetas)

                # etiquetar al hijo
                etiquetar(nombre_siguiente_nodo, indice_etiqueta_actual, grafo)

                # sumar en uno el nombre para el siguiente nodo
                nombre_siguiente_nodo = nombre_siguiente_nodo + 1

        print(f"\tNumero de nodos en esta generacion: {len(dict_generaciones[n])} ")

    # comprobar que sea un DAG
    assert nx.is_directed_acyclic_graph(grafo)

    # convertirlas etiquetas en colores
    # tambien ponerle colores al grafo
    for nodo in grafo.nodes():
        grafo.nodes[nodo]['color'] = etiquetas[get_etiqueta(nodo, grafo)]

    # devolver el grafo y el diccionario de generaciones
    return grafo, dict_generaciones


def generacion_multitype_bp(transition_matrix, offspring_distributions, etiquetas, k):
    # se le dan los parametros que denifen un multitype branching procces
    # estos son: transition_matrix, offspring_distributions y las etiquetas
    # se le da un vector de generacion

    # Se asume que Z_(n-1) = k
    # Se calcula Z_n

    # poner la generacion creada en un vector
    # logicamente comienza en 0
    z = np.zeros_like(k)

    # obtener los indices de las etiquetas
    idx_etiquetas = [idx for idx, e in enumerate(etiquetas)]

    # comprobar que k considere todos los tipos
    assert len(k) == len(idx_etiquetas)

    # generar la proxima generacion
    # iterar en los tipos
    for tipo in idx_etiquetas:

        # obtener las distribuciones de este tipo
        # para numero de hijos y para sus tipos
        offspring_padre = offspring_distributions[tipo]
        proba_etiquetas = transition_matrix[tipo]

        # repetir el proceso
        # segun cuantas unidades de este tipo haya en k
        for _ in range(k[tipo]):

            # ver cuantos hijos va a tener
            num_hijos = sample_from_offspring_distribution(offspring_padre)

            # dados los hijos, la distribucion sobre los vectores
            # de descendencia se  distribuye multinomial
            descendencia = np.random.multinomial(num_hijos, proba_etiquetas)

            # añadir su descendencia
            # a la descendencia de la generacion
            z = z + descendencia

    # devolver la generacion nueva
    return z


def muestra_generacion_multitype_bp(transition_matrix, offspring_distributions, etiquetas, k, num_samples):
    # se le dan los parametros que denifen un multitype branching procces
    # estos son: transition_matrix, offspring_distributions y las etiquetas
    # se le da un vector de generacion

    # Se asume que Z_(n-1) = k
    # Se calcula Z_n num_samples veces

    # se devuelve una matriz donde las filas son los vectores k sampleados

    muestra = []

    # iterar en el numero de elementos que se quieren
    for _ in range(num_samples):

        # agregar un elementos a la muestra
        muestra.append(generacion_multitype_bp(transition_matrix=transition_matrix,
                        offspring_distributions=offspring_distributions,
                        etiquetas=etiquetas, k=k))

    # hacerlo matriz y devolver
    muestra = np.array(muestra)
    return muestra



## -----------------------------------------------------------------------------------------------------------------------------

# MULTI-TYPE BRANCHING PROCESS CON DISTRIBUCION POISSON

# una funcion que toma un parametros lambda
# regresa una muestra de un elemento
# de una distribucion poison con ese parametro
def sample_from_poisson(lambda_):

    # Generate a random sample from the Poisson distribution
    poisson_sample = np.random.poisson(lambda_)

    return poisson_sample


def generar_multi_bp_poisson(numero_de_generaciones, transition_matrix, lamba_vector, etiquetas, indice_etiqueta_raiz):
    # genera el arbol y lo va etiquetando con el modelo
    # de multi-type branching process
    # donde el numero de hijos de cada tipo es una poisson

    # numero_de_generaciones -  cuantas generaciones poner en el arbol
    # transition_matrix - probabilidade de cambiar de tipo a tipo
    # lamba_vector - parametros lambdas de las distribuciones poisson de cada tipo
    # etiquetas - los colores asociados a los indices de las etiquetas
    # indice_etiqueta_raiz - como etiquetar al nodo raiz

    # crear el grafo
    grafo = nx.DiGraph()

    # hacer un diccionario de generaciones, es decir de pares
    # numero_de_generacion : [nombre de los nodos en esa generacion]
    dict_generaciones = dict()

    # los nodos tienen nombres de numeros
    # ir llevando esta variable
    # sumar en uno cada que se pone un nuevo nodo
    nombre_siguiente_nodo = 0

    # poner la 0-esima generacion
    grafo.add_node(nombre_siguiente_nodo)

    # indicarlo en el diccionario de generaciones
    dict_generaciones[0] = [nombre_siguiente_nodo]

    # ponerle la etiquera correspondiente al nodo raiz
    etiquetar(nombre_siguiente_nodo, indice_etiqueta_raiz, grafo)

    # sumar en uno el nombre del siguiente nodo
    nombre_siguiente_nodo = nombre_siguiente_nodo + 1

    # poner tantas generaciones como se quieran
    for n in range(1, numero_de_generaciones + 1):

        print(f"Creando y etiquetando la generacion: {n}")

        # indicar que en este generacion,
        # de entrada no se tienen nodos
        # inicializarlo de esta forma
        dict_generaciones[n] = []

        # iterar en los nodos de la generacion pasada
        for node_padre in dict_generaciones[n-1]:

            # el numero de hijos, y tipo de estos,
            # dependen del tipo del padre
            tipo_padre = get_etiqueta(node_padre, grafo)

            # ver cual es la offspring distribution correspondiente
            # segun el tipo del padre
            # para esto, tomar el parametro lambda de su distribucion poisson
            lambda_padre = lamba_vector[tipo_padre]

            # usar esta distribucion poisson con este parametro
            # para delimitar el numero de hijos
            num_hijos = sample_from_poisson(lambda_padre)

            # poner cada uno de estos hijos
            for _ in range(num_hijos):

                # poner un nodo extra, que es el hijo
                grafo.add_node(nombre_siguiente_nodo)

                # poner la conexion de padre a hijo
                grafo.add_edge(node_padre, nombre_siguiente_nodo)

                # añadir al hijo a los nodos de esta generacion
                dict_generaciones[n].append(nombre_siguiente_nodo)

                # etiquetar al hijo
                # esto depende de la etiqueta del padre

                # tomas las probabilidades
                proba_etiquetas = transition_matrix[tipo_padre]

                # hacer un sampleo, segun estas probabilidades
                # para seleccionar el indice de esta etiqueta
                indice_etiqueta_actual = elegir_indice_con_probabilidades(proba_etiquetas)

                # etiquetar al hijo
                etiquetar(nombre_siguiente_nodo, indice_etiqueta_actual, grafo)

                # sumar en uno el nombre para el siguiente nodo
                nombre_siguiente_nodo = nombre_siguiente_nodo + 1

        print(f"\tNumero de nodos en esta generacion: {len(dict_generaciones[n])} ")

    # comprobar que sea un DAG
    assert nx.is_directed_acyclic_graph(grafo)

    # convertirlas etiquetas en colores
    # tambien ponerle colores al grafo
    for nodo in grafo.nodes():
        grafo.nodes[nodo]['color'] = etiquetas[get_etiqueta(nodo, grafo)]

    # devolver el grafo y el diccionario de generaciones
    return grafo, dict_generaciones


## -----------------------------------------------------------------------------------------------------------------------------


# OBTENER PROPORCION DE etiquetas

def obtener_proporciones_etiquetas(lista_etiquetas, etiquetas):

    # obtener las proporciones de las etiquetas

    # comprobar que se tena el mismo numero de etiquetas
    assert len(set(lista_etiquetas)) == len(etiquetas)

    # proporciones[0] = proporcion de elementos con etiqueta con indice 0

    proporciones = [-1] * len(etiquetas)

    # iterar en las etiquetas, mas bien, sus indices
    for idx_etiqueta in range(len(etiquetas)):

        # obtener el numero de individuos con este indice de etiqueta
        num_elementos = sum([1 for etiqueta_ind in lista_etiquetas
                             if etiqueta_ind == idx_etiqueta])

        # añadir la proporcion
        # el numero de individuos con ese etiqeuta
        # sobre el numero total de individuos
        proporciones[idx_etiqueta] = num_elementos/len(lista_etiquetas)

    # comprobar que suma a 1
    assert np.isclose(np.sum(proporciones), 1)

    return proporciones


## -----------------------------------------------------------------------------------------------------------------------------
