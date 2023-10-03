# scp41.txt

# format:
# no. of rows (m), no. of columns (n)
# cost of each column c(j)
# for each row i, the no of columns which cover row i
# followed by a list of the columns which cover row i

import numpy as np

# Prompt the user to enter a file name
# filename = input("Enter the file name: ")

# Print the user's input
# print("You entered:", filename)

# Include file type extension (.txt)
# filename = filename + ".txt"

# f = open(filename, "r")

# Reading the text file line by line
f = open("scp510.txt", "r")
content = f.readlines()

# Extracting the no. of rows and no. of columns info from first line
dim = content[0].split()
m = int(dim[0])
n = int(dim[1])

# Creating empty matrices of the correct size
c = np.empty((n, 1))
A = np.empty((m, n))
x = np.zeros((n, 1))

# Indexers
row_count = 1
j = 0

# Filling the cost (c) array
for row in content[1:]:
    row_str = row.split()
    row_int = [int(string) for string in row_str]
    for elem in row_int:
        c[j][0] = elem
        j += 1

    row_count += 1

    if j >= n:
        break

# Filling the relevant rows of the A matrix with 1s in the designated columns
row_num = -1

for row in content[row_count:]:
    row_str = row.split()
    row_int = [int(string) for string in row_str]
    if len(row_int) == 1 and row_int[0] <= m:
        row_num += 1
        continue

    for elem in row_int:
        col_num = elem
        A[row_num][col_num-1] = 1

# Check A matrix

# Calculate the sum across each row
row_sums = A.sum(axis=1)
# Find the rows where the sum is zero
zero_sum_rows = np.where(row_sums == 0)[0]
# Count the number of rows where the sum is zero
num_zero_sum_rows = zero_sum_rows.shape[0]
# print("Number of rows with zero sum:", num_zero_sum_rows)

# Calculate the sum across each column
col_sums = A.sum(axis=0)
# Find the columns where the sum is zero
zero_sum_cols = np.where(col_sums == 0)[0]
# Count the number of rows where the sum is zero
num_zero_sum_cols = zero_sum_cols.shape[0]
# print("Number of columns with zero sum:", num_zero_sum_cols)