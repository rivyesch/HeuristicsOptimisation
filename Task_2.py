from Task_1 import c, A, x

import numpy as np
import time

# # Simple Test Case
# c = np.array([[40], [30], [20], [50], [25], [70], [40], [80], [100], [20], [25]])
# A = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#               [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#               [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
# x = np.zeros((11, 1))

# To find the objective function value (i.e. cost) of the solution
def totalCost(cost, solution):
    totalCost = np.dot(cost.transpose(), solution)[0]
    return totalCost

# Finds the number of columns selected (optional and not necessary)
def coverLength(solution):
    coverLength = len(np.where(solution >= 1)[0])
    return coverLength

def greedy_search(cost, constraint, solution, start=time.time()):

    U = constraint.copy()  # Duplicate A matrix
    first_iteration = True
    counter = 0  # No. of iterations through the loop
    covered = set()
    mask = np.ones(U.shape[0], dtype=bool)

    # Stops when all the constraints have been satisfied (no. of constraints is the no. of rows in U)
    while len(covered) < U.shape[0]:

        if first_iteration:
            zero_indices = np.where(cost == 0)[0]
            if len(zero_indices) > 0:
                for zero_col in zero_indices:
                    matching_rows = U[:, zero_col] == 1  # Find rows where the zero cost column has a value of 1
                    matching_rows = np.where(matching_rows)[0]
                    U = np.delete(U, matching_rows, axis=0)  # Remove the row from U matrix
                    solution[zero_col] = 1
            first_iteration = False

        mask[list(covered)] = False  # To avoid going back to columns that have already been covered (covered indexes are set to false)
        col_sums = np.sum(U[mask], axis=0)  # Sum of each column (excluding those that have already been covered)

        ratios = cost.transpose() / col_sums  # Divide by corresponding c(j)
        min_ratio_ind = np.nanargmin(ratios)  # Index of smallest ratio (not zero and not NaN)

        solution[min_ratio_ind] = 1  # Updates the solution (replaces 0 with 1)

        lhs = np.dot(U, solution)  # Multiply U and x using numpy.dot()
        one_indices = np.where(lhs >= 1)[0]  # Index where lhs meets constraint (>= 1)

        covered |= set(one_indices)

        # Update the counters and sets maximum number of iterations before loop is terminated
        counter += 1
        if counter == 100:
            print("Too many iterations!")
            break

    print("Constraints satisfied: ", len(np.where(np.dot(constraint, solution) >= 1)[0]), "/", U.shape[0])
    print("No. of columns selected: ", coverLength(solution))
    print("No. of columns not selected: ", len(np.where(solution < 1)[0]))
    print("Column selected (indexes) are:")
    print(np.where(solution == 1)[0])
    print()
    print("Initial solution (cost) is:", totalCost(cost, solution))
    print('time: ', time.time() - start, "seconds\n")

    return solution


initial_sol = greedy_search(c, A, x)
