from Task_1 import c, A, x
# from Task_2 import c, A, x
from Task_2 import initial_sol, totalCost, coverLength

import numpy as np
import time


def checkConstraints(constraint, solution):
    lhs = np.dot(constraint, solution)

    # Find the rows that sum up to zero
    sums = constraint.sum(axis=1)
    zero_rows = np.where(sums == 0)[0]
    # Remove the zero rows
    lhs_reduced = np.delete(lhs, zero_rows, axis=0)

    # Check if all element is greater than or equal to 1 (outputs True if satisfied)
    constraint_satisfied = np.all(lhs_reduced >= 1)

    return constraint_satisfied

# Finds all neighbours to the solution
# The index positions of zeros and ones in the solution are first identified
# Then swaps are done between zeros and ones for all possible swap combinations
# List of neighbours that satisfy the constraint are returned
def getNeighbours(constraint, solution):
    neighbours = []
    x_ones_indices = np.where(solution == 1)[0]
    x_zero_indices = np.where(solution == 0)[0]
    for i in x_ones_indices:
        for j in x_zero_indices:
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]

            # Neighbours still need to be a correct solution that satisfies the constraint
            if checkConstraints(constraint, neighbour):
                neighbours.append(neighbour)

    print("No. of neighbours: ", len(neighbours))
    return neighbours

def getBestNeighbour(cost, constraint, neighbours):
    # Initialising by setting the first neighbour to be best
    bestTotalCost = totalCost(cost, neighbours[0])
    bestNeighbour = neighbours[0]

    for neighbour in neighbours:
        # Calculates the cost of each neighbour
        currentTotalCost = totalCost(cost, neighbour)
        if currentTotalCost < bestTotalCost:
            # Updates the best neighbour if the cost is lower
            bestTotalCost = currentTotalCost
            bestNeighbour = neighbour

    return bestNeighbour, bestTotalCost

def hillClimbing(cost, constraint, solution, start=time.time()):
    # Initialise with a solution (greedy search sol)
    currentSolution = solution
    currentTotalCost = totalCost(cost, currentSolution)

    neighbours = getNeighbours(constraint, currentSolution)
    bestNeighbour, bestNeighbourTotalCost = getBestNeighbour(cost, constraint, neighbours)

    # Updates the best solution and cost while it is smaller (better) than the current cost
    while bestNeighbourTotalCost < currentTotalCost:
        currentSolution = bestNeighbour
        currentTotalCost = bestNeighbourTotalCost

        # Finds neigbourhood and best neighbour
        neighbours = getNeighbours(constraint, currentSolution)
        bestNeighbour, bestNeighbourTotalCost = getBestNeighbour(cost, constraint, neighbours)

    print("Constraints satisfied: ", len(np.where(np.dot(constraint, bestNeighbour) >= 1)[0]), "/", A.shape[0])
    print("No. of columns selected: ", coverLength(bestNeighbour))
    print("No. of columns not selected: ", len(np.where(bestNeighbour < 1)[0]))
    print("Column selected (indexes) are:")
    print(np.where(bestNeighbour == 1)[0])
    print()
    print("Hill climb solution (cost) is: ", bestNeighbourTotalCost)
    print('time: ', time.time() - start, "seconds\n")

    return currentSolution, currentTotalCost


if __name__ == "__main__":
    # code to be executed when running this file as the main program
    hillClimbing(c, A, initial_sol)