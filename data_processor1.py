import numpy as np
import pandas as pd
import random
from data_loader import load_csv

def calculate_mean(data):
    return np.mean(data, axis=0)

def calculate_std(data):
    return np.std(data, axis=0)

def identify_outliers(data, mean, std, threshold=2):
    outliers = []
    for row in data:
        if any(abs(value - mean[i]) > threshold * std[i] for i, value in enumerate(row)):
            outliers.append(row)
    return outliers

def remove_outliers(data, mean, std, threshold=2):
    filtered_data = []
    for row in data:
        if all(abs(value - mean[i]) <= threshold * std[i] for i, value in enumerate(row)):
            filtered_data.append(row)
    return filtered_data

def normalize_data(data, mean, std):
    normalized_data = (data - mean) / std
    normalized_data = 2 * (normalized_data - np.min(normalized_data, axis=0)) / (np.max(normalized_data, axis=0) - np.min(normalized_data, axis=0)) - 1
    return normalized_data

def main():
    # Load data from Q1.csv and Q2.csv
    q1_data = np.array(load_csv("Q1.csv"))
    q2_data = np.array(load_csv("Q2.csv"))

    # Calculate mean and standard deviation for each column
    q1_mean = calculate_mean(q1_data)
    q1_std = calculate_std(q1_data)
    q2_mean = calculate_mean(q2_data)
    q2_std = calculate_std(q2_data)

    # Identify and display outliers
    q1_outliers = identify_outliers(q1_data, q1_mean, q1_std)
    q2_outliers = identify_outliers(q2_data, q2_mean, q2_std)
    print(f"Q1.csv outliers: {q1_outliers}")
    print(f"Q2.csv outliers: {q2_outliers}")

    # Remove outliers
    q1_filtered = remove_outliers(q1_data, q1_mean, q1_std)
    q2_filtered = remove_outliers(q2_data, q2_mean, q2_std)

    # Normalize data
    q1_normalized = normalize_data(np.array(q1_filtered), q1_mean, q1_std)
    q2_normalized = normalize_data(np.array(q2_filtered), q2_mean, q2_std)

    # Display 10 randomly selected values from the normalized data
    print("10 random values from Q1 normalized data:", random.sample(list(q1_normalized), 10))
    print("10 random values from Q2 normalized data:", random.sample(list(q2_normalized), 10))

    # Load normalized data into Panda DataFrames and NumPy arrays
    q1_df = pd.DataFrame(q1_normalized, columns=["y", "x"])
    q2_df = pd.DataFrame(q2_normalized, columns=["y", "x1", "x2"])
    q1_np = np.array(q1_normalized)
    q2_np = np.array(q2_normalized)

    # Display 10 randomly selected values from each DataFrame and NumPy array
    print("10 random values from Q1 DataFrame:", q1_df.sample(10))
    print("10 random values from Q2 DataFrame:", q2_df.sample(10))
    print("10 random values from Q1 NumPy array:", q1_np[np.random.choice(q1_np.shape[0], 10, replace=False)])
    print("10 random values from Q2 NumPy array:", q2_np[np.random.choice(q2_np.shape[0], 10, replace=False)])

    # Split data into X_data and Y_data
    X_data_q1 = q1_np[:, 1]
    Y_data_q1 = q1_np[:, 0]
    X_data_q2 = q2_np[:, 1:]
    Y_data_q2 = q2_np[:, 0]

    # Save X_data and Y_data
    np.save("X_data_q1.npy", X_data_q1)
    np.save("Y_data_q1.npy", Y_data_q1)
    np.save("X_data_q2.npy", X_data_q2)
    np.save("Y_data_q2.npy", Y_data_q2)

if __name__ == "__main__":
    main()