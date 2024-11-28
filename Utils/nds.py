import numpy as np

def non_dominated_sort(population, objectives):
    """Perform non-dominated sorting on the population.

    Args:
        population (list): List of solutions in the population.
        objectives (np.ndarray): Array of objective function values for each solution.

    Returns:
        list: List of fronts where each front is a list of indices of solutions.
    """
    num_solutions = len(population)
    dominance_counts = np.zeros(num_solutions)
    dominated_solutions = [set() for _ in range(num_solutions)]
    fronts = [[]]

    # Compute dominance counts and dominated solutions
    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            # Check dominance relationship
            if (objectives[i] <= objectives[j]).all() and (objectives[i] < objectives[j]).any():
                # i dominates j
                dominated_solutions[i].add(j)
                dominance_counts[j] += 1
            elif (objectives[j] <= objectives[i]).all() and (objectives[j] < objectives[i]).any():
                # j dominates i
                dominated_solutions[j].add(i)
                dominance_counts[i] += 1

    # Identify the first front (solutions with zero dominance counts)
    for i in range(num_solutions):
        if dominance_counts[i] == 0:
            fronts[0].append(i)

    # Iterate through each front
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for idx in fronts[current_front]:
            for dominated in dominated_solutions[idx]:
                dominance_counts[dominated] -= 1
                if dominance_counts[dominated] == 0:
                    next_front.append(dominated)
        current_front += 1
        if next_front:
            fronts.append(next_front)

    return fronts

def calculate_crowding_distance(objectives, front):
    """Calculate the crowding distance for each solution in a front.

    Args:
        objectives (np.ndarray): Array of objective function values.
        front (list): List of indices of solutions in a particular front.

    Returns:
        np.ndarray: Array of crowding distances for each solution in the front.
    """
    num_solutions = len(front)
    num_objectives = objectives.shape[1]
    distances = np.zeros(num_solutions)

    # Calculate crowding distance for each objective
    for m in range(num_objectives):
        # Sort solutions in the front by objective m
        sorted_indices = np.argsort(objectives[front, m])
        sorted_front = [front[i] for i in sorted_indices]

        # Set infinity distance for boundary solutions
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf

        # Calculate distances for remaining solutions
        for i in range(1, num_solutions - 1):
            distances[sorted_indices[i]] += (
                objectives[sorted_front[i + 1], m] - objectives[sorted_front[i - 1], m]
            )

    return distances

def select_population(population, objectives, population_size):
    """Perform the environment selection based on NSGA-II.

    Args:
        population (list): List of solutions in the population.
        objectives (np.ndarray): Array of objective function values for each solution.
        population_size (int): Desired size of the selected population.

    Returns:
        list: List of selected solutions based on NSGA-II selection.
    """
    # Perform non-dominated sorting
    fronts = non_dominated_sort(population, objectives)

    # List to store the selected solutions
    selected_population = []
    front_index = 0

    # Select solutions based on fronts
    while len(selected_population) + len(fronts[front_index]) <= population_size:
        # Calculate crowding distance for the current front
        front = fronts[front_index]
        crowding_distances = calculate_crowding_distance(objectives, front)

        # Pair solutions with their crowding distance and sort by distance (descending)
        front_with_distances = list(zip(front, crowding_distances))
        front_with_distances.sort(key=lambda x: x[1], reverse=True)

        # Add sorted solutions to the selected population
        for idx, _ in front_with_distances:
            selected_population.append(population[idx])

        # Move to the next front
        front_index += 1

    # If the population size is exceeded, perform a partial selection from the last front
    if len(selected_population) < population_size:
        front = fronts[front_index]
        crowding_distances = calculate_crowding_distance(objectives, front)
        front_with_distances = list(zip(front, crowding_distances))
        front_with_distances.sort(key=lambda x: x[1], reverse=True)

        # Add the required number of solutions from the current front
        for idx, _ in front_with_distances[:population_size - len(selected_population)]:
            selected_population.append(population[idx])

    return selected_population

# Example usage:
if __name__ == "__main__":
    # Example population and objectives (for demonstration)
    # This is just a simple example; you should replace it with your actual data.
    population = [
        [0.1, 0.5],
        [0.3, 0.4],
        [0.2, 0.6],
        [0.6, 0.2],
        [0.4, 0.8],
        [0.8, 0.3],
    ]
    objectives = np.array([
        [0.1, 0.5],
        [0.3, 0.4],
        [0.2, 0.6],
        [0.6, 0.2],
        [0.4, 0.8],
        [0.8, 0.3],
    ])
    population_size = 4

    # Perform environment selection
    selected_population = select_population(population, objectives, population_size)

    # Output the selected population
    print("Selected population:")
    for individual in selected_population:
        print(individual)
