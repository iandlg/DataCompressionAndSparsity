import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

train_arrays = np.load("./processed_data/train_arrays.npy")
train_labels = np.load("./processed_data/train_labels.npy")
test_arrays = np.load("./processed_data/test_arrays.npy")
test_labels = np.load("./processed_data/test_labels.npy")

# List of row sizes for the projection matrices
rows_list = [25, 100, 200, 500]
cols = 784  # Image vector dimension

# Generate random matrices and normalize columns
projection_matrices = []
for rows in rows_list:
    W = np.random.randn(rows, cols)  # Draw from standard normal distribution
    W /= np.linalg.norm(W, axis=0, keepdims=True)  # Normalize columns to unit L2 norm
    projection_matrices.append(W)

# Project the train and test images onto the matrices
projected_train = [W @ train_arrays.T for W in projection_matrices]  # Shape: (rows, num_samples)
projected_test = [W @ test_arrays.T for W in projection_matrices]  # Shape: (rows, num_samples)

# Convert projected data back to (num_samples, rows)
projected_train = [p.T for p in projected_train]
projected_test = [p.T for p in projected_test]

# Print the resulting shapes
for i, rows in enumerate(rows_list):
    print(f"Projection with {rows} rows -> Train shape: {projected_train[i].shape}, Test shape: {projected_test[i].shape}")
