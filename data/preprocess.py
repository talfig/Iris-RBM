import os
import numpy as np
from sklearn.datasets import load_iris


def preprocess_iris(output_path, percentile_file_path, binary_output_path):
    """
    Preprocess the Iris dataset for RBM training and save it as a NumPy file,
    dividing the dataset into small, medium, and large categories based on petal length,
    and also save the percentile values for each feature in a structured numpy array.
    Save both the binary data and the regular data with one-hot encoded labels.

    Parameters:
        output_path (str): Path to save the preprocessed regular data.
        percentile_file_path (str): Path to save the percentile values as a numpy file.
        binary_output_path (str): Path to save the binary transformed data.

    Returns:
        None
    """
    # Load the Iris dataset
    iris = load_iris()
    features = iris.data  # Shape: (150, 4)
    labels = iris.target  # Shape: (150,)

    # Calculate percentiles for each column (feature) in the dataset
    percentiles = np.array([
        np.percentile(features[:, 0], [33, 66]),
        np.percentile(features[:, 1], [33, 66]),
        np.percentile(features[:, 2], [33, 66]),
        np.percentile(features[:, 3], [33, 66])
    ])

    # Save percentiles as a numpy file (rows: feature index, columns: 33rd and 66th percentiles)
    os.makedirs(os.path.dirname(percentile_file_path), exist_ok=True)
    np.save(percentile_file_path, percentiles)

    print(f"Percentile values saved to {percentile_file_path}")

    # Transform features to binary representation based on percentiles
    binary_features = transform_to_binary(features, percentiles)
    one_hot_labels = one_hot_encode_labels(labels)

    # Save the binary features, one-hot labels, and percentiles as a .npz file
    os.makedirs(os.path.dirname(binary_output_path), exist_ok=True)
    np.savez_compressed(binary_output_path, features=binary_features, labels=one_hot_labels, percentiles=percentiles)

    print(f"Binary preprocessed Iris dataset saved to {binary_output_path}")

    # Save the regular features, one-hot labels, and percentiles as a .npz file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, features=features, labels=one_hot_labels, percentiles=percentiles)

    print(f"Regular preprocessed Iris dataset saved to {output_path}")


def transform_vector_to_binary(vector, percentiles):
    """
    Transform a single feature vector to binary values based on the percentiles.

    Parameters:
        vector (tuple): A feature vector (e.g., (5.2, 1.4, 3.5, 1.3)).
        percentiles (numpy.ndarray): The percentiles for each feature.

    Returns:
        list: A list of binary values representing the vector.
    """
    binary_vector = []

    # Iterate over each feature in the vector
    for i, value in enumerate(vector):
        # Get the 33rd and 66th percentiles for the current feature
        p33, p66 = percentiles[i]

        # Transform value to binary: small(1), medium(2), large(3)
        if value <= p33:
            binary_vector.extend([1, 0, 0])  # Small
        elif value <= p66:
            binary_vector.extend([0, 1, 0])  # Medium
        else:
            binary_vector.extend([0, 0, 1])  # Large

    return binary_vector


def transform_to_binary(features, percentiles):
    """
    Transform the features of the Iris dataset into binary values based on the 33rd and 66th percentiles.

    Parameters:
        features (numpy.ndarray): The feature matrix.
        percentiles (numpy.ndarray): The percentiles for each feature.

    Returns:
        numpy.ndarray: The binary transformed feature matrix.
    """
    binary_features = []

    # Iterate over each sample in the features
    for sample in features:
        # Use the transform_vector_to_binary function to convert each sample
        binary_sample = transform_vector_to_binary(sample, percentiles)

        # Append the transformed binary sample to the list
        binary_features.append(binary_sample)

    return np.array(binary_features)


def one_hot_encode_labels(labels):
    """
    Convert integer labels into one-hot encoded format.

    Parameters:
        labels (numpy.ndarray): The labels array (e.g., [0, 1, 2, 0, 1, ...]).

    Returns:
        numpy.ndarray: The one-hot encoded labels.
    """
    # Number of classes (for Iris, there are 3 classes)
    num_classes = len(np.unique(labels))

    # One-hot encoding
    one_hot_labels = np.eye(num_classes)[labels]

    return one_hot_labels


if __name__ == "__main__":
    output_path = "iris/preprocessed_iris.npz"
    percentile_file_path = "iris/percentiles.npy"  # Saving percentiles as a NumPy file
    binary_output_path = "iris/binary_preprocessed_iris.npz"  # New path for binary data
    preprocess_iris(output_path, percentile_file_path, binary_output_path)
