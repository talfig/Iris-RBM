import numpy as np


def loader(file_path):
    """
    Load the preprocessed Iris dataset from a .npz file.

    Parameters:
        file_path (str): Path to the .npz file containing the preprocessed data.

    Returns:
        tuple: Loaded features and labels ready for training.
    """
    data = np.load(file_path)
    features = data["features"]
    labels = data["labels"]  # Load labels
    print(f"Loaded preprocessed Iris dataset from {file_path}")
    return features, labels


if __name__ == "__main__":
    file_path = "iris/binary_preprocessed_iris.npz"
    iris_features, iris_labels = loader(file_path)
    print("Features shape:", iris_features.shape)
    print("Labels shape:", iris_labels.shape)
