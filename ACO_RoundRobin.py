import random
import numpy as np

class ACO_Scheduler:
    def __init__(self, num_tasks, num_nodes, iterations, alpha=1, beta=2, evaporation_rate=0.5):
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.iterations = iterations
        self.alpha = alpha  # Influência dos feromônios
        self.beta = beta    # Influência da heurística
        self.evaporation_rate = evaporation_rate
        self.pheromones = np.ones((num_tasks, num_nodes))  # Inicializando feromônios
        self.heuristic = np.random.rand(num_tasks, num_nodes)  # Heurística aleatória

    def schedule(self):
        best_solution = None
        best_cost = float('inf')
        
        for _ in range(self.iterations):
            solutions = []
            costs = []
            
            for _ in range(self.num_tasks):
                probabilities = (self.pheromones ** self.alpha) * (self.heuristic ** self.beta)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                solution = [np.random.choice(self.num_nodes, p=probabilities[i]) for i in range(self.num_tasks)]
                cost = self.evaluate_solution(solution)
                solutions.append(solution)
                costs.append(cost)
            
            # Atualiza melhor solução
            min_cost_index = np.argmin(costs)
            if costs[min_cost_index] < best_cost:
                best_cost = costs[min_cost_index]
                best_solution = solutions[min_cost_index]
            
            # Atualiza feromônios
            self.pheromones *= (1 - self.evaporation_rate)
            for i, node in enumerate(best_solution):
                self.pheromones[i][node] += 1.0 / best_cost  # Refórço positivo
                
        return best_solution

    def evaluate_solution(self, solution):
        return sum(solution)  # Exemplo de custo (pode ser modificado)


class RoundRobin_Scheduler:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.current_node = 0
    
    def schedule(self, tasks):
        schedule = {}
        for task in tasks:
            schedule[task] = self.current_node
            self.current_node = (self.current_node + 1) % self.num_nodes
        return schedule


# Exemplo de uso
if __name__ == "__main__":
    num_tasks = 10
    num_nodes = 3

    print("\nEscalonamento com Ant Colony Optimization:")
    aco = ACO_Scheduler(num_tasks, num_nodes, iterations=100)
    best_aco_solution = aco.schedule()
    print("Tarefas escalonadas nos nós:", best_aco_solution)
    
    print("\nEscalonamento com Round Robin:")
    rr = RoundRobin_Scheduler(num_nodes)
    rr_solution = rr.schedule(list(range(num_tasks)))
    print("Tarefas escalonadas nos nós:", rr_solution)
