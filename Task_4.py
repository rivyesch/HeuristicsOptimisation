# Tabu Search uses local search but can accept a worse solution in order to prevent getting stuck in local minimums.
# to avoid getting stuck in local minima. The basic idea of Tabu Search is to penalize moves that take the solution into
# previously visited search spaces

# Short term memory: prevent from revisiting previously visited solution and also used to return to good components in
# order to localise and intensify a search
# Long term memory: diversify the search and explore unvisited areas of the search space by avoiding explored areas

# Tabu Tenure: no. of iterations a move stays in Tabu list
# Asiration Criteria:
# Frequency Memory: total no. of iterations that each solution was picked since the beginning of the search.
# Solutions that were visited more are less likely to be picked again and would promote more diverse solutions.

from Task_1 import c, A, x
from Task_2 import initial_sol, totalCost, coverLength
from Task_3 import checkConstraints, getNeighbours

import numpy as np
import time

def getTabuStructure(constraint, solution):
    dict = {}

    x_ones_indices = np.where(solution == 1)[0]
    x_zero_indices = np.where(solution == 0)[0]

    z_grid, o_grid = np.meshgrid(x_zero_indices, x_ones_indices)
    comb = np.column_stack((z_grid.ravel(), o_grid.ravel()))

    allCombinations = (tuple(map(tuple, comb)))

    for swap in allCombinations:
        dict[swap] = {'tabu_time': 0, 'MoveValue': 0, 'freq': 0, 'Penalized_MV': 0}

    print("All combinations: ", len(dict))

    return dict

def swapMove(solution, i, j):

    neighbour = solution.copy()
    neighbour[i] = solution[j]
    neighbour[j] = solution[i]

    return neighbour

# The algorithm is being tested with tabu tenure 5 with Penalization weight set to 0.5
# Termination condition is set to 3 consecutive iterations with no best solution found
# An aspiration criterion set to when the objective function value is better than the best known one
def tabuSearch(cost, constraint, solution, tenure, Penalization_weight, start=time.time()):
    initial_objvalue = totalCost(cost, solution)
    tabu_structure = getTabuStructure(constraint, solution)
    best_solution = solution
    best_objvalue = totalCost(cost, best_solution)
    current_solution = solution
    current_objvalue = totalCost(cost, current_solution)

    iter = 1
    Terminate = 0

    while Terminate < 3:

        # Search the neighbourhood of the current solution:
        for move in tabu_structure:
            candidate_solution = swapMove(current_solution, move[0], move[1])
            candidate_objvalue = totalCost(cost, candidate_solution)
            tabu_structure[move]['MoveValue'] = candidate_objvalue
            # Penalized objValue by simply adding freq to Objvalue (minimization):
            tabu_structure[move]['Penalized_MV'] = candidate_objvalue + (tabu_structure[move]['freq'] * Penalization_weight)

        # Filter based on Move_Value to remove those solutions worse than the initial method
        filtered_tabu = {k: v for k, v in tabu_structure.items() if initial_objvalue - 10 < v['MoveValue'] <= initial_objvalue}
        tabu_structure = filtered_tabu.copy()

        # Remove those neighbourhoods that done satisfy all constraints
        for key in filtered_tabu:
            i = key[0]
            j = key[1]
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]

            if not checkConstraints(constraint, neighbour):
                del tabu_structure[(i, j)]

        print("Valid combinations: ", len(tabu_structure))

        # Admissible move
        while True:
            # Select the move with the lowest Penalised Objective Function Value (total cost) in the neighbourhood
            best_move = min(tabu_structure, key=lambda x: tabu_structure[x]['Penalized_MV'])
            MoveValue = tabu_structure[best_move]["MoveValue"]
            tabu_time = tabu_structure[best_move]["tabu_time"]

            # For non-Tabu
            if tabu_time < iter:
                # Make the move
                current_solution = swapMove(current_solution, best_move[0], best_move[1])
                current_objvalue = totalCost(cost, current_solution)

                # Best Improving Move
                if MoveValue < best_objvalue:
                    if checkConstraints(constraint, current_solution):
                        best_solution = current_solution
                        best_objvalue = current_objvalue
                    Terminate = 0
                else:
                    Terminate += 1

                # Update dictionary for that particular best candidate neighbour
                tabu_structure[best_move]['tabu_time'] = iter + tenure
                tabu_structure[best_move]['freq'] += 1
                iter += 1
                break

            # For those neighbours in Tabu (list)
            else:
                # Aspiration criterion : if better the best solution we accept regardless if tabu or not
                if MoveValue < best_objvalue:
                    # Make the move
                    current_solution = swapMove(current_solution, best_move[0], best_move[1])
                    current_objvalue = totalCost(cost, current_solution)
                    best_solution = current_solution
                    best_objvalue = current_objvalue
                    tabu_structure[best_move]['freq'] += 1
                    Terminate = 0
                    iter += 1
                    break
                else:
                    # To prevent cycling back to this same neighbour repeatedly
                    tabu_structure[best_move]['Penalized_MV'] = float('inf')
                    continue

    print("Constraints satisfied: ", len(np.where(np.dot(constraint, best_solution) >= 1)[0]), "/", constraint.shape[0])
    print("No. of columns not selected: ", len(np.where(best_solution < 1)[0]))
    print("Column selected (indexes) are:")
    print(np.where(best_solution == 1)[0])
    print('time: ', time.time() - start, "seconds\n")

    tabuSearchRunTime = time.time() - start

    return best_solution, best_objvalue, tabuSearchRunTime

tabuSearch(c, A, initial_sol)
# tabuList = set()
# tabuList.append(bestNeighbour)
# if len(tabulist) > tabuSize:
#     tabuList.pop(0)



