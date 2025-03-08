import time
import numpy as np
from PythonExercise.data_loader import load_csv

def main():
    # Load x values from L1.csv and Q1.csv into Python arrays
    l1_data = load_csv("L1.csv")
    q1_data = load_csv("Q1.csv")
    x_l1 = [row[1] for row in l1_data]
    x_q1 = [row[1] for row in q1_data]

    # Perform element-wise multiplication using a Python loop
    start_time = time.time()
    dot_product_python = sum(x_l1[i] * x_q1[i] for i in range(len(x_l1)))
    end_time = time.time()
    print(f"Dot product (Python loop): {dot_product_python}")
    print(f"Time taken (Python loop): {end_time - start_time} seconds")

    # Load x values into NumPy arrays
    x_l1_np = np.array(x_l1)
    x_q1_np = np.array(x_q1)

    # Perform element-wise multiplication using NumPy
    start_time = time.time()
    dot_product_numpy = np.dot(x_l1_np, x_q1_np)
    end_time = time.time()
    print(f"Dot product (NumPy): {dot_product_numpy}")
    print(f"Time taken (NumPy): {end_time - start_time} seconds")

    # Load data from Q2.csv into Python arrays and NumPy arrays
    q2_data = load_csv("Q2.csv")
    q2_python = [[row[1], row[2]] for row in q2_data]
    q2_numpy = np.array(q2_python)

    # Compute the square of the arrays (matrix multiplication) using Python loop
    start_time = time.time()
    q2_python_squared = [[sum(a * b for a, b in zip(row, col)) for col in zip(*q2_python)] for row in q2_python]
    end_time = time.time()
    print(f"Time taken for matrix multiplication (Python loop): {end_time - start_time} seconds")

    # Compute the square of the arrays (matrix multiplication) using NumPy
    start_time = time.time()
    q2_numpy_squared = np.dot(q2_numpy, q2_numpy.T)
    end_time = time.time()
    print(f"Time taken for matrix multiplication (NumPy): {end_time - start_time} seconds")

if __name__ == "__main__":
    main()