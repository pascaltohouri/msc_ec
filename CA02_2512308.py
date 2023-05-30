#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !!! for colab or jupyter notebook only !!!
#!pip install --upgrade networkx decorator
#!pip install networkx
#!pip install deap


import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

# Question 2: This question demands function whose input is an NxN array (the adjacency matrix). F1 outputs the average path length, F2 outputs the diameter, F3 outputs the total number of links, F4 checks the graph is connected.

def floyd_warshall(adj_matrix):                                             # define a function for obtaining the shortest path lengths between node pairs
    n = len(adj_matrix)                                                     # define n as the number of nodes in the graph
    dist = np.array(adj_matrix, dtype=float)                                # define a new array called dist as the adjacency matrix

    for i in range(n):                                                      # Initialize the distances between non-adjacent vertices as infinity
        for j in range(n):
            if i != j and dist[i][j] == 0:
                dist[i][j] = float('inf')

    for k in range(n):                                                      # sequentially replace the d_ij by checking for shorter routes through k
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

def f1_average_path_length(adj_matrix):                                     # find the arithmetic mean of the path length matrix
    n = len(adj_matrix)
    
    dist = floyd_warshall(adj_matrix)
    total_paths = n * (n - 1)
    total_length = np.sum(dist) - np.sum(np.diag(dist))                     # Exclude the diagonal elements (self-loops)
    
    avg_length = total_length / total_paths
    return avg_length
                                                
def f2_diameter(adj_matrix):                                                # Use a breadth-first algorithm
    def bfs(adj_matrix, start):
        visited = set()
        queue = [(start, 0)]
        max_distance = 0
        while queue:
            vertex, distance = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                max_distance = max(max_distance, distance)
                for neighbor in range(len(adj_matrix)):
                    if adj_matrix[vertex][neighbor] == 1:
                        queue.append((neighbor, distance + 1))
        return max_distance

    max_distance = 0
    for vertex in range(len(adj_matrix)):
        max_distance = max(max_distance, bfs(adj_matrix, vertex))
    return max_distance

def f3_count_edges(adj_matrix):                                             # define a variable 'count'=0. for each row in the adjacency matrix, sum the 1's in each row and add the sum to 'count', to obtain undirected edges, half count.
    count = 0
    for row in adj_matrix:
        count += sum(row)
    return count // 2

def f4_is_connected(adj_matrix):                                            # Use a depth-first algorithm
    def dfs(vertex, visited):
        visited.add(vertex)
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[vertex][neighbor] == 1 and neighbor not in visited:
                dfs(neighbor, visited)

    visited = set()
    dfs(0, visited)
    return len(visited) == len(adj_matrix)
    

# Question 3: F5 adds a link between two randomly chosen nodes, F6 removes a link between two randomly chosen nodes, F7 combines F5 and F5.

def f5_add_link(adj_matrix):                                                     # Count the nodes 'n', define the max edges n(n-1)/2, if the sum of the row sums is n(n-1)/2, the graph is fully connected. Leave it alone!
    n = len(adj_matrix)
    max_edges = n * (n - 1) // 2
    current_edges = sum(sum(row) for row in adj_matrix) // 2
    if current_edges == max_edges:
        print("The graph is fully connected. No new links added.")
        return adj_matrix

    while True:                                                                  # While i,j is within 0,0 and n-1,n-1: if i is not j and the two are not directly connected, edit the adjacency matrix s.t. they become connected.
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j and adj_matrix[i][j] == 0:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
            print(f"Added link between vertices {i} and {j}.")
            break
    return adj_matrix

def f6_remove_link(adj_matrix):                                                  # create a list, add edge i,j until the list contains all the edges in the graph.
    n = len(adj_matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))

    random.shuffle(edges)                                                       # Select  "edge index" to select an edge i,j and remove it from the list. If the f4_is_connected function returns True, print the edge removed and the new adjacency matrix. Otherwise add the links back.     
    for edge in edges:
        i, j = edge
        adj_matrix[i][j] = 0
        adj_matrix[j][i] = 0
        if f4_is_connected(adj_matrix):
            print(f"Removed link between vertices {i} and {j}.")
            return adj_matrix
        else:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
    return adj_matrix 

def f7_random_rewiring(adj_matrix):
    f5_add_link(adj_matrix)
    f6_remove_link(adj_matrix)
    return adj_matrix


# Question 4: The fitness function is the average path length of the graph. Smaller average path lengths are preferred, so this evolutionary algorithm selects the minimal average path length. In each iteration, the graph receives random rewiring via function f7. 

def f8_random_graph_operation(adj_matrix, p1, p2):
    p3 = 1 - p1 - p2
    operation = random.choices(
        [f5_add_link, f6_remove_link, f7_random_rewiring],
        weights=[p1, p2, p3],
        k=1
    )[0]
    return operation(np.copy(adj_matrix))

def fitness_func_q4(adj_matrix):
    return f2_diameter(adj_matrix) - f1_average_path_length(adj_matrix)


def evolutionary_algorithm(adj_matrix, population_size, num_generations, fitness_func, p1, p2):    # define an evolutionary algorithm with inputs 'adj_matrix', 'population_size', 'num_generations', 'F' - fitness function, probabilities 'p1,p2'
    population = [np.copy(adj_matrix) for n in range(population_size)]                  # copy 'adj_matrix' 'population_size' times
    for generation in range(num_generations):                                           # an iteration is a generation. For each 
        print("Generation:", generation)
        new_population = []
        for i in range(population_size):                                                # for each member of the current generation
            new_individual = f8_random_graph_operation(population[i], p1, p2)           # randomly mutate
            new_population.append(new_individual)                                       # add the new individual to the new_population

        population = new_population                                                     # the new population becomes the starting point for the next mutations
        fitnesses = [fitness_func(solution) for solution in population]                            # ***create a list displaying the fitness for each solution in the population
        sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])      # ***given the length of the fitnesses list, an corresponding index 'k', create a sorted indices list 
        
        population = [population[i] for i in sorted_indices]                            # ***make a new list of the sorted population based on the sorted indices
        population = population[:-5]                                                    # Remove the 5 solutions with the highest fitness
        population.extend([np.copy(solution) for solution in population[:5]])           # Replicate the 5 solutions with the lowest fitness

    best_solutions = population[:3]
    return best_solutions


def draw_graph(adj_matrix):
    G = nx.Graph()
    for i in range(len(adj_matrix)):                                                    # Add edges based on the adjacency matrix
        for j in range(i+1, len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    pos = nx.circular_layout(G)                                                         # Draw the graph using a circular layout
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            font_weight='bold')    
    plt.show()                                                                          # Show the graph  
    

# Question 5a 

def fitness_func_q5a(adj_matrix, a1, a2, a3):
    return a1 * f1_average_path_length(adj_matrix) + a2 * f2_diameter(adj_matrix) + a3 * f3_count_edges(adj_matrix)

def fitness_func_q5b(adj_matrix, a1, a2, a3):
    return a1 * f1_average_path_length(adj_matrix) + a2 * f2_diameter(adj_matrix) + a3 * f3_count_edges(adj_matrix)

def dominates(p, q):                                                
    for i in range(len(p)):                                                            # Loop through each element in both input lists (p and q)                                         
        if p[i] > q[i]:                                                                # If the element in p is greater than the corresponding element in q,
           return False                                                               # p does not dominate q; return False
    return True

def calculate_crowding_distance(fronts):
    distance = [0 for front in range(len(fronts))]                                     # Initialize the crowding distance array with zeros
    sorted_front = sorted(fronts, key=lambda x: x[0])                                  # Sort the front based on the first objective
    distance[0] = float('inf')                                                         # Assign infinite crowding distance to the first and last solutions in the sorted front
    distance[len(fronts) - 1] = float('inf')
    
    for i in range(1, len(fronts) - 1):                                                # Calculate crowding distance for the remaining solutions based on the first objective
        distance[i] += sorted_front[i + 1][0] - sorted_front[i - 1][0]
    
    for m in range(1, len(front[0])):                                                  # Iterate through the other objectives (m) and update the crowding distance
        sorted_front = sorted(fronts, key=lambda x: x[m])                               # Sort the front based on the current objective m
        
        distance[0] = float('inf')                                                     # Assign infinite crowding distance to the first and last solutions in the sorted front
        distance[len(front) - 1] = float('inf')
        
        for i in range(1, len(front) - 1):                                             # Update the crowding distance for the remaining solutions based on the current objective m
            distance[i] += sorted_front[i + 1][m] - sorted_front[i - 1][m]
    
    return distance

def fast_non_dominated_sort(fit_comp):
    fronts = []                                                                       # Initialize empty list for Pareto fronts
    S = [[] for values in range(len(fit_comp))]                                       # Initialize empty lists for domination sets (S) and domination count (n) for each solution
    n = [0 for values in range(len(fit_comp))]
    rank = [0 for values in range(len(fit_comp))]                                     # Initialize rank list for each solution

    for p in range(len(fit_comp)):                                                   # Loop through all solutions and compare them
        S[p] = []
        n[p] = 0
        for q in range(len(fit_comp)):
            if dominates(fit_comp[p], fit_comp[q]):
                if q not in S[p]:
                    S[p].append(q)                                                     # Add q to the domination set of p
            elif dominates(fit_comp[q], fit_comp[p]):
                n[p] += 1                                                              # Increase the domination count of p
        if n[p] == 0:                                                                  # If the domination count of p is 0, it belongs to the first front (rank 0)
            rank[p] = 0
            if len(fronts) == 0:
                fronts.append([])                                    
            fronts[0].append(p)                                      

    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if len(Q) == 0:
                        Q.append(q)
                    else:
                        Q.append(q)
        i += 1
        fronts.append(Q)

    del fronts[len(fronts) - 1]                                                        # Remove the last empty front
    return fronts                                                                      # Return the list of Pareto fronts


def nsga2(adj_matrix, population_size, num_generations, fitness_func, p1, p2, a1, a2, a3):         # define an evolutionary algorithm with inputs 'adj_matrix', 'population_size', 'num_generations', 'F' - fitness function, probabilities 'p1,p2'
    
    population = [np.copy(adj_matrix) for n in range(population_size)]                 # copy 'adj_matrix' 'population_size' times
    for generation in range(num_generations):                                          # an iteration is a generation. For each generation print the generation number and an empty list
        print("Generation:", generation)
        new_population = []
        for i in range(population_size):                                               # for each member of the current generation
            new_individual = f8_random_graph_operation(population[i], p1, p2)          # randomly mutate
            new_population.append(new_individual)                                      # add the new individual to the new_population

        population = new_population                                                    # the new population becomes the starting point for the next mutations
        fitnesses = [fitness_func(solution, a1, a2, a3) for solution in population]    # create a list displaying the fitness for each solution in the population
        sorted_indices = sorted(range(len(fitnesses)), key=lambda k: fitnesses[k])     # given the length of the fitnesses list, an corresponding index 'k', create a sorted indices list 
        
        population = [population[i] for i in sorted_indices]
        population = population[:-5]
        population.extend([np.copy(solution) for solution in population[:5]])
        
        
        fit_comp = [[f1_average_path_length(i), f2_diameter(i), f3_count_edges(i)] for i in population]
        fronts = fast_non_dominated_sort(fit_comp)
        print(fronts)
        best_solutions = []
        
        for front in fronts:
            if len(best_solutions) + len(front) <= 10:                                             # Change the stopping condition to 10 instead of 3
                best_solutions.extend([population[i] for i in front])                              # Add the top 10 population indices to best solutions
            else:
                remaining_spots = 10 - len(best_solutions)                                         # Change the remaining spots calculation to 10 instead of 3
                front_objective_values = [fitnesses[i] for i in front]
                crowding_distance = calculate_crowding_distance(front_objective_values)
                index_to_crowding_distance = {front[i]: crowding_distance[i] for i in range(len(front))}
                sorted_front = sorted(front, key=lambda x: index_to_crowding_distance[x], reverse=True)
                best_solutions.extend([population[i] for i in sorted_front[:remaining_spots]])
                break
    print(best_solutions)
    #best_solutions_fitness_values = [fitness_func(solution, a1, a2, a3) for solution in best_solutions]
    return best_solutions



def draw_graph(adj_matrix):
    G = nx.Graph()
    for i in range(len(adj_matrix)):
        for j in range(i+1, len(adj_matrix[i])):
            value = adj_matrix[i][j]
            if isinstance(value, np.ndarray):
                if value.size != 1:
                    print(f"Problematic value at ({i}, {j}): {value}")  # Print the problematic value and its location
                value = value.item(0) if value.size == 1 else 0

            if value == 1:
                G.add_edge(i, j)

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            font_weight='bold')
    plt.show()

 


    

########################################################## TEST #############################################################


adj_matrix = [                                                        # ring graph N=10
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
]


draw_graph(adj_matrix)

# Uncomment as needed #
#### Question 2 ####
print(f1_average_path_length(adj_matrix)) 
print(f2_diameter(adj_matrix))
print(f3_count_edges(adj_matrix))
print(f4_is_connected(adj_matrix))

#### Question 3 ####
#print(f5_add_link(adj_matrix))
#print(f6_remove_link(adj_matrix))
#print(f7_random_rewiring(adj_matrix))

#### Question 4 ####
# Parameters
#p1 = 0.4
#p2 = 0.4
#population_size = 50
#num_generations = 1000
#fitness_func = fitness_func_q4

#best_solutions = evolutionary_algorithm(adj_matrix, population_size, num_generations, fitness_func, p1, p2)
#for solution in best_solutions:
    #draw_graph(solution)
    #print(solution)
    #print(f1_average_path_length(solution))
    #print(f2_diameter(solution))
    #print(f2_diameter(solution)-f1_average_path_length(solution))

#### Question 5 ####
# Parameters
#p1 = 0.33
#p2 = 0.33
#population_size = 50
#num_generations = 1000
#fitness_func = fitness_func_q5a

# run 1: 
#a1 = 10
#a2 = 5
#a3 = 1

# run 2: 
#a1 = 5
#a2 = 10
#a3 = 1

# run 3: 
#a1 = 1
#a2 = 5
#a3 = 10

#best_solutions = evolutionary_algorithm(adj_matrix, population_size, num_generations, lambda x: fitness_func(x, a1, a2, a3), p1, p2)
#for solution in best_solutions:
    #draw_graph(solution)
    #print(solution)
    #print(f1_average_path_length(solution))
    #print(f2_diameter(solution))

#### Question 5b ####
# Parameters
#p1 = 0.33
#p2 = 0.33
#population_size = 50
#num_generations = 100
#fitness_func = fitness_func_q5b

#a1 = 1
#a2 = 1
#a3 = 1

#pareto_front = nsga2(adj_matrix, population_size, num_generations, fitness_func, p1, p2, a1, a2, a3)
#display_results(pareto_front)
#for solution in pareto_front:
    #draw_graph(solution)
    
#pareto_front_fitness_values = [fitness_func_q5(solution, a1, a2, a3) for solution in best_solutions_fitness_values]
#plot_pareto_fronts(pareto_front_fitness_values)

