## using ml_env

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# Download and load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

# Convert training data to NumPy arrays
train_images = np.array([np.array(img[0]) for img in train_dataset]).squeeze()  # Extract images
train_labels = np.array([img[1] for img in train_dataset]).squeeze()  # Extract labels

# Convert test data to NumPy arrays
test_images = np.array([np.array(img[0]) for img in test_dataset]).squeeze()  # Extract images
test_labels = np.array([img[1] for img in test_dataset]).squeeze()  # Extract labels

# Sparsify data
threshold = 0.1
train_images = np.where(train_images<threshold,0,train_images) # check if we should put 1 or the non zero val
test_images = np.where(test_images<threshold,0,test_images)

plt.imshow(train_images[500,:,:],cmap='gray')
plt.show()

# Print shapes
print("Train Images Shape:", train_images.shape)  # (60000, 28, 28)
print("Train Labels Shape:", train_labels.shape)  # (60000,)
print("Test Images Shape:", test_images.shape)    # (10000, 28, 28)
print("Test Labels Shape:", test_labels.shape)    # (10000,)

# Reshape images to vectors
rows = train_images.shape[1]
cols = train_images.shape[2]
train_num = train_images.shape[0]
test_num = test_images.shape[0]

train_arrays = train_images.reshape(train_num, rows*cols)
test_arrays = test_images.reshape(test_num, rows*cols)


print("Train Array Shape:", train_arrays.shape)  # (60000, 784)
print("Test Array Shape:", test_arrays.shape)    # (10000, 784)

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


np.save("./processed_data/train_arrays", train_arrays, )
np.save("./processed_data/test_arrays", test_arrays)
np.save("./processed_data/train_labels", train_labels)
np.save("./processed_data/test_labels", test_labels)
